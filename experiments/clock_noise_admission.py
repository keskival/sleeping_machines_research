"""Source-frozen numerical admission and bounded MG DEV fit; run_safe only.

Imports no numerical runtime until source/data/prerequisite/physical guards pass.
Existing public queues and core kernels are not modified. No official tau17.
"""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import resource
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
from public_speech_admission import require_guard_environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_from_increments(last_observed, warm_increment, increments):
    """Mirror the native FP32 feedback addition, including the warm prediction.

    A cumsum followed by adding the base has a different rounding order from
    the actual feedback stream. Keep each realized FP32 addition here.
    """
    def fp32(value):
        return struct.unpack('<f', struct.pack('<f', value))[0]
    value = fp32(fp32(last_observed) + fp32(warm_increment))
    predictions = []
    for delta in increments:
        value = fp32(value + fp32(delta))
        predictions.append(value)
    return predictions


def preflight(path, stage, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Changed frozen manifest')
    manifest = json.loads(Path(path).read_text())
    if manifest['status'] != 'prepared_unrun' or stage not in ('contracts', 'pilot'):
        raise ValueError('Frozen admission stage required')
    for name, expected in manifest['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed frozen source: ' + name)
    cfg = manifest['stages'][stage]
    if Path(cfg['tag']).name != cfg['tag'] or (ROOT / cfg['output']).exists():
        raise ValueError('Unique unused plain tag required')
    if list((ROOT / cfg['output']).parent.glob(cfg['tag'] + '*.pt')):
        raise ValueError('Preserve interrupted checkpoints; use a uniquely tagged retry')
    if stage == 'pilot':
        parent = json.loads((ROOT / manifest['stages']['contracts']['output']).read_text())
        if (parent['status'] != 'completed' or parent['stage'] != 'contracts' or
                parent['source_sha256'] != manifest['source_sha256'] or parent['manifest_sha256'] != sha(path) or
                parent['config'] != manifest['stages']['contracts']):
            raise ValueError('Completed exact native contracts required')
        data = manifest['data']
        if data['tau'] != 19 or sha(ROOT / data['path']) != data['sha256']:
            raise ValueError('Frozen official tau19 DEV data required')
        if parent['max_rss_kb'] * 1.5 > cfg['rss_cap_kb']:
            raise ValueError('Measured contract RSS needs a revised guard')
    return manifest, cfg


def gradients(model):
    return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n, p in model.named_parameters()}


def close(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=2e-8, atol=2e-9)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            close(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            close(a, b)
    else:
        assert first == second


def contracts(cfg):
    rows = [dict(events=[(float(t), [(-1.) ** t, .2 * t, 1.]) for t in range(length)]) for length in (4, 2)]
    torch.manual_seed(71)
    base = fast_class(AddressedEventHeads)(sources=1, content_dim=3, classes=2, payload=4, depth=3, heads=2, pool=3).double()
    checks = []
    for temperature in cfg['temperatures']:
        outputs = []
        for backend in ('custom', 'layer'):
            model = copy.deepcopy(base)
            model.zero_grad(set_to_none=True)
            function = clock_batched_logits if backend == 'custom' else clock_compiled_logits
            z = function(model, rows, 73, temperature=temperature, all_logits=True, route_credit='linear')
            loss = F.cross_entropy(z[:, 0], torch.tensor([0, 1])) + z.square().mean()
            loss.backward()
            outputs.append((z.detach().clone(), gradients(model)))
            plain = function(model, rows, 73, temperature=temperature, all_logits=True, route_credit=None)
            torch.testing.assert_close(plain, z.detach(), rtol=0, atol=0)
        close(*outputs)
        checks.append(f'nu={temperature}: independent custom/layer logits, every gradient and forward-credit invariance')
        if temperature == 1.:
            model = copy.deepcopy(base)
            z = batched_logits(model, rows, 73, all_logits=True, route_credit='linear')
            loss = F.cross_entropy(z[:, 0], torch.tensor([0, 1])) + z.square().mean()
            loss.backward()
            torch.testing.assert_close(z, outputs[0][0], rtol=0, atol=0)
            for name, gradient in gradients(model).items():
                torch.testing.assert_close(gradient, outputs[0][1][name], rtol=0, atol=0)
            checks.append('nu=1: exact native batched endpoint, every parameter gradient')
    # Actual compilation at one controlled shape; no full-data allocation.
    m = copy.deepcopy(base)
    z = clock_compiled_logits(m, rows, 73, temperature=.5, compile_step=True, all_logits=True, route_credit='linear')
    (F.cross_entropy(z[:, 0], torch.tensor([0, 1])) + z.square().mean()).backward()
    reference = copy.deepcopy(base)
    rz = clock_batched_logits(reference, rows, 73, temperature=.5, all_logits=True, route_credit='linear')
    (F.cross_entropy(rz[:, 0], torch.tensor([0, 1])) + rz.square().mean()).backward()
    close((z, gradients(m)), (rz, gradients(reference)))
    checks.append('nu=.5: actual compiled logits and every gradient match custom native program')
    for temperature in cfg['temperatures']:
        fitted = []
        for backend in ('custom', 'compiled'):
            model = copy.deepcopy(base).float()
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            histories = []
            for seed in (76, 77):
                optimizer.zero_grad(set_to_none=True)
                if backend == 'custom':
                    logits = clock_batched_logits(model, rows, seed, temperature=temperature, route_credit='linear')
                else:
                    logits = clock_compiled_logits(model, rows, seed, temperature=temperature, compile_step=True, route_credit='linear')
                F.cross_entropy(logits, torch.tensor([0, 1])).backward()
                histories.append((logits.detach().clone(), gradients(model)))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            fitted.append((histories, model.state_dict()))
        for (za, ga), (zb, gb) in zip(fitted[0][0], fitted[1][0]):
            torch.testing.assert_close(za, zb, rtol=2e-4, atol=2e-5)
            for name in ga:
                torch.testing.assert_close(ga[name], gb[name], rtol=2e-4, atol=2e-5)
        for name in fitted[0][1]:
            torch.testing.assert_close(fitted[0][1][name], fitted[1][1][name], rtol=2e-4, atol=2e-5)
        checks.append(f'nu={temperature}: FP32 actual compiled/custom every gradient and two Adam updates within declared tolerances')
    # Exact recovery after an actual optimizer update, before the next update.
    for temperature in cfg['temperatures']:
        model = copy.deepcopy(base)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        def update(m, opt, seed):
            opt.zero_grad(set_to_none=True)
            z = clock_batched_logits(m, rows, seed, temperature=temperature, route_credit='linear')
            F.cross_entropy(z, torch.tensor([0, 1])).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        update(model, optimizer, 80)
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        restored = copy.deepcopy(base)
        restored.load_state_dict(saved['model'])
        ropt = torch.optim.Adam(restored.parameters(), lr=.003)
        ropt.load_state_dict(saved['optimizer'])
        update(model, optimizer, 81)
        update(restored, ropt, 81)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        close(optimizer.state_dict(), ropt.state_dict())
        checks.append(f'nu={temperature}: exact restored next Adam weights; every moment matches')
    return dict(checks=checks, scope='Native numerical port/recovery admission, not benchmark quality or full expected-risk gradient correctness.')


def pilot(cfg, manifest, directory):
    import numpy as np
    from mackey_glass_native import taps_of, smape
    from race_language_screen import capture
    raw = np.load(ROOT / manifest['data']['path'])
    offset = int(37.5 * cfg['repeat'])
    raw = raw[offset:offset + 1501].astype(np.float64)
    if len(raw) != 1501:
        raise ValueError('Official DEV series is too short')
    mu, sd = raw[:751].mean(), raw[:751].std()
    z = ((raw - mu) / sd).astype(np.float32)
    results = []
    for temperature in cfg['temperatures']:
        started = time.perf_counter()
        torch.manual_seed(cfg['seed'])
        rng = np.random.default_rng(cfg['seed'] + 1)
        model = fast_class(AddressedEventHeads)(sources=1, content_dim=8, classes=1, payload=16, depth=2, heads=2, pool=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        losses, work = [], None
        for step in range(cfg['steps']):
            starts = rng.integers(0, 750 - cfg['segment'] + 1, cfg['lanes'])
            rows = [dict(events=[(float(i), taps_of(z, i, 8)) for i in range(s, s + cfg['segment'])]) for s in starts]
            y = torch.tensor(np.stack([z[s + 1:s + cfg['segment'] + 1] - z[s:s + cfg['segment']] for s in starts]))
            box = {}
            def update(trace=False):
                optimizer.zero_grad(set_to_none=True)
                out = clock_compiled_logits(model, rows, 1000 + step, temperature=temperature,
                                            compile_step=not trace, all_logits=True, route_credit='linear')[..., 0]
                loss = F.mse_loss(out[:, 8:], y[:, 8:])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite integrated pilot loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                box['loss'] = float(loss.detach())
            if step == 0:
                work = capture(lambda: update(trace=True))
                if not work['formula_coverage_complete']: raise ValueError('Incomplete first-window fitting work')
            else:
                update()
            losses.append(box['loss'])
        checkpoint = directory / (cfg['tag'] + '_nu' + str(temperature).replace('.', 'p') + '.pt')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
                        temperature=temperature, source_sha256=manifest['source_sha256'], data=manifest['data'],
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), checkpoint)
        model.eval()
        rows = [dict(events=[(float(t), taps_of(z, t, 8) if t < 750 else np.zeros(8, np.float32)) for t in range(1500)])]
        feedback = (750, lambda prev, logits: torch.cat([prev[:, :1] + logits[:, :1].to(prev.dtype), prev[:, :-1]], -1))
        with torch.no_grad():
            rollout = clock_compiled_logits(model, rows, 777, temperature=temperature, compile_step=True,
                all_logits=True, feedback=feedback)[0, :, 0]
        predicted = np.array(forecast_from_increments(float(z[749]), rollout[749].item(), rollout[750:].tolist()))
        score = smape(predicted * sd + mu, raw[751:1501])
        per = (work['arithmetic_flops'] + work['special_function_evaluations']) / (cfg['lanes'] * (cfg['segment'] - 8))
        results.append(dict(temperature=temperature, dev_smape=score, first_train_mse=losses[0], final_train_mse=losses[-1],
                            checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                            parameters=sum(p.numel() for p in model.parameters()),
                            fitting_targets=cfg['steps'] * cfg['lanes'] * (cfg['segment'] - 8),
                            whole_fit_flops_estimate=per * cfg['steps'] * cfg['lanes'] * (cfg['segment'] - 8),
                            fit_flops_per_presented_target_estimate=per, first_window_work=work,
                            wall_s=time.perf_counter() - started, forecast_finite=bool(np.isfinite(predicted).all())))
        print(json.dumps({k: v for k, v in results[-1].items() if k != 'first_window_work'}), flush=True)
    return dict(arms=results, scope='Single tau19 DEV repeat, bounded integration fit, all proposals evaluated. Same initialized model/sampling seeds; future trajectories diverge. First full eager update extrapolated; compilation/evaluation/checkpoint/setup/traffic/energy extra. No sparse inference saving, official tau17 score or promotion from this tiny fit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--stage', choices=('contracts', 'pilot'), required=True)
    args = parser.parse_args()
    manifest, cfg = preflight(args.manifest, args.stage, args.manifest_sha256)
    if Path('/.dockerenv').exists():
        raise ValueError('Workspace container has no physical-host reservation')
    require_guard_environment(cfg)
    global torch, F, fast_class, AddressedEventHeads, batched_logits, clock_batched_logits, clock_compiled_logits
    import torch
    from torch.nn import functional as F
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.clock_noise_episodes import clock_batched_logits, clock_compiled_logits
    from torch._inductor import config as inductor_config
    torch.set_num_threads(1)
    inductor_config.compile_threads = 1
    started = time.perf_counter()
    output = ROOT / cfg['output']
    output.parent.mkdir(parents=True, exist_ok=True)
    result = contracts(cfg) if args.stage == 'contracts' else pilot(cfg, manifest, output.parent)
    result.update(status='completed', stage=args.stage, config=cfg, manifest_sha256=sha(args.manifest),
                  source_sha256=manifest['source_sha256'], wall_s=time.perf_counter() - started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1))
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
