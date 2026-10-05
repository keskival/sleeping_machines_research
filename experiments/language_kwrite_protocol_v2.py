"""Guarded k-write protocol repair; all fitted/traced/evaluated steps use one policy.

Separate from the original queued wrapper. No numerical imports before physical
run_safe admission. The manifest freezes sources and arguments for each stage.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(manifest_path, stage):
    manifest = json.loads(Path(manifest_path).read_text())
    cfg = manifest['stages'][stage]
    if manifest['status'] != 'prepared_unrun':
        raise ValueError('Prepared frozen protocol required')
    for path, expected in manifest['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Changed frozen source: ' + path)
    for prior in cfg['requires']:
        previous = manifest['stages'][prior]
        record = json.loads((ROOT / previous['output']).read_text())
        if (record.get('status') != 'completed' or record.get('stage') != prior
                or record.get('manifest_sha256') != digest(manifest_path)):
            raise ValueError('Passed exact-source predecessor required: ' + prior)
        if previous['kind'] == 'smoke' and record['max_rss_kb'] * 1.5 > cfg['guards']['MEM_CAP_RSS_KB']:
            raise ValueError('Measured smoke needs a larger fresh resource reservation')
    for parent in cfg.get('external_requires', []):
        record = json.loads((ROOT / parent['output']).read_text())
        if record.get('status') != 'completed':
            raise ValueError('Completed external comparison required')
        if any(record['args'].get(k) != v for k, v in parent['expected_args'].items()):
            raise ValueError('External comparison arguments differ')
        work = record['work']['whole_fit_unit_special_flops_estimate']
        score = record['test_bpc_eval_segment']
        if (not math.isfinite(work) or not 0 < work <= parent['cap_flops'] or
                not math.isfinite(score) or score >= parent['reference_bpc'] or
                record.get('eval_segment') != 256 or record.get('test_targets_eval_segment') != 999936):
            raise ValueError('External result does not pass the frozen win gate')
        if any(manifest['source_sha256'].get(k) != v
               for k, v in record['source_sha256'].items()):
            raise ValueError('External numerical producer differs from confirmation')
    if (ROOT / cfg['output']).exists():
        raise ValueError('Preserve existing stage result')
    if Path('/.dockerenv').exists():
        raise ValueError('Review container has no physical-host reservation')
    if socket.gethostname() != manifest['host']:
        raise ValueError('Stage belongs to its declared physical host')
    guards = cfg['guards']
    if guards['MIN_AVAIL_MB'] < 8192:
        raise ValueError('At least 8GiB available memory required')
    if any(os.environ.get(k) != str(v) for k, v in guards.items()):
        raise ValueError('Explicit frozen run_safe guards required')
    if any(os.environ.get(k) != '1' for k in
           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCH_NUM_THREADS')):
        raise ValueError('One thread required')
    lock = '/tmp/experiments-runner.lock'
    slot = os.environ.get('AWS_GYM_SLOT')
    if slot:
        if socket.gethostname() != 'ip-172-31-47-132' or slot not in ('1', '2', '3'):
            raise ValueError('Unauthorized bounded slot')
        lock = '/tmp/experiments-runner.aws-gym-slot' + slot + '.lock'
    if os.readlink('/proc/self/fd/9') != lock:
        raise ValueError('Inherited run_safe reservation required')
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return manifest, cfg


def install_policy(base, ce, recruit_layer, torch, k, deliver):
    """Use the same layer for traced optimizer steps and compiled train/eval."""
    def eager_step(*args):
        return recruit_layer(*args, write_k=k, deliver_k=deliver)[:7]

    cache = {}

    def compiled_step():
        if 'step' not in cache:
            cache['step'] = torch.compile(eager_step, dynamic=False, fullgraph=True)
        return cache['step']

    def eager_logits(model, rows, seed, all_logits=False, route_credit=None):
        return ce.compiled_logits(model, rows, seed, all_logits=all_logits,
                                  route_credit=route_credit, step=eager_step)

    ce.compiled_step = compiled_step
    # The base driver calls this explicit eager path for its first traced updates.
    base.batched_logits = eager_logits
    return eager_step, compiled_step


def install_accounting(base):
    from race_language_screen import RaceAudit
    traces = []

    class KWriteAudit(RaceAudit):
        def formula(self, func, args, kwargs, out):
            if str(func).split('.')[1].rstrip('_') == 'topk':
                x, k = args[:2]
                dim = args[2] if len(args) > 2 else kwargs.get('dim', -1)
                width = x.shape[dim]
                rows = x.numel() // width
                return (0, 0, rows * k * (width - 1),
                        'top-k comparison proxy (k linear scans); not measured device work')
            return super().formula(func, args, kwargs, out)

    def capture(action):
        with KWriteAudit() as audit:
            action()
        record = audit.result()
        if not record['formula_coverage_complete']:
            raise ValueError(record['unsupported_floating_operators'])
        record['exponential_random_draws'] = sum(
            r['floating_output_elements'] for r in record['operators'].values()
            if r['classification'].startswith('RNG generation'))
        traces.append(record)
        return record

    base.capture = capture
    return traces


def contracts(base, ce, recruit_layer, torch):
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    from sleeping_machines.batched_episodes import batched_logits
    original_layer = ce.layer_step
    rows = [dict(events=[(float(t), [float(t % 3 == j) for j in range(3)])
                         for t in range(length)]) for length in (5, 3)]
    records = []
    for k in (1, 2):
        torch.manual_seed(91)
        model = fast_class(AddressedEventHeads)(sources=1, content_dim=3, classes=3,
                                               payload=4, depth=2, heads=2, pool=4)
        step, compiled = install_policy(base, ce, recruit_layer, torch, k, False)

        def checked_step(*args):
            result = step(*args)
            inactive = ~args[5]
            for after, before in zip(result[2:5], args[2:5]):
                torch.testing.assert_close(after[inactive], before[inactive], atol=0, rtol=0)
            if not args[4].any():
                expected = k * args[5][:, None].expand(-1, model.heads)
                torch.testing.assert_close(result[4].sum(-1), expected, atol=0, rtol=0)
            return result

        eager = ce.compiled_logits(model, rows, 17, all_logits=True,
                                   route_credit='linear', step=checked_step)
        compiled_z = ce.compiled_logits(model, rows, 17, all_logits=True,
                                        route_credit='linear', step=compiled())
        torch.testing.assert_close(eager, compiled_z, atol=3e-5, rtol=3e-5)
        params = tuple(model.parameters())
        a = torch.autograd.grad(eager.square().sum(), params, allow_unused=True)
        b = torch.autograd.grad(compiled_z.square().sum(), params, allow_unused=True)
        for x, y in zip(a, b):
            if (x is None) != (y is None):
                raise ValueError('Gradient support differs')
            if x is not None:
                torch.testing.assert_close(x, y, atol=3e-4, rtol=3e-4)
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError('Nonfinite contract gradient')
        if k == 1:
            reference = batched_logits(model, rows, 17, all_logits=True, route_credit='linear')
            torch.testing.assert_close(eager, reference, atol=3e-5, rtol=3e-5)
            ref_grad = torch.autograd.grad(reference.square().sum(), params, allow_unused=True)
            for x, y in zip(a, ref_grad):
                if (x is None) != (y is None):
                    raise ValueError('k1/base gradient support differs')
                if x is not None:
                    torch.testing.assert_close(x, y, atol=3e-4, rtol=3e-4)
            reference2 = ce.compiled_logits(model, rows, 17, all_logits=True,
                                           route_credit='linear', step=original_layer)
            torch.testing.assert_close(eager, reference2, atol=0, rtol=0)
        records.append(dict(write_k=k, eager_compiled_logits=True,
                            every_parameter_gradient=True, unequal_length_padding=True,
                            inactive_state_unchanged=True, first_event_write_count=True))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--stage', required=True)
    args = p.parse_args()
    manifest, cfg = preflight(args.manifest, args.stage)
    started = time.monotonic()
    sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
    import torch
    from torch._inductor import config as inductor_config
    from torch._dynamo import config as dynamo_config
    import sleeping_machines.compiled_episodes as ce
    from sleeping_machines.recruit_layer import recruit_layer
    import language_batched_benchmark as base
    torch.set_num_threads(1)
    inductor_config.compile_threads = 1
    dynamo_config.cache_size_limit = 64
    out = ROOT / cfg['output']
    if cfg['kind'] == 'contracts':
        result = dict(status='completed', contracts=contracts(base, ce, recruit_layer, torch))
    else:
        install_policy(base, ce, recruit_layer, torch, cfg['write_k'], cfg['deliver_k'])
        traces = install_accounting(base)
        sys.argv = [str(ROOT / 'experiments/language_batched_benchmark.py'), *cfg['arguments']]
        base.main()
        result = json.loads(out.read_text())
        result.update(write_k=cfg['write_k'], deliver_k=cfg['deliver_k'])
        result['protocol']['write_policy'] = 'identical k-write policy for traced updates, training and evaluation'
        result['work']['scope'] = ('Full eager k-write forward/loss/backward/clip/Adam traced on first windows; '
                                  'extrapolated over fit. Additional top-k selection and state movement are '
                                  'not physical runtime/traffic measurements. Inference work not established.')
        result['work']['traces'] = traces
        result['benchmark_eligible'] = cfg['kind'] == 'fit'
    result.update(stage=args.stage, manifest_sha256=digest(args.manifest),
                  source_sha256=manifest['source_sha256'])
    result['max_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.setdefault('wall_s', time.monotonic() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], stage=args.stage, output=cfg['output'])))


if __name__ == '__main__':
    main()
