"""Guarded integrated positive-time-scale learning and continuation contracts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--tag', required=True)
    cli = p.parse_args()
    packet = json.loads(Path(cli.manifest).read_text())
    pins = packet['contract_source_sha256']
    def check():
        for name, digest in pins.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    check()
    out = ROOT / 'experiments/results/diagnostics' / (cli.tag + '.json')
    if out.exists(): raise FileExistsError(out)
    import torch
    from aws_private_bank_model import PrivateBankModel
    from aws_private_bank_time_model import TimeScaledPrivateBankModel
    from sleeping_machines.paired_route_credit import paired_route_credit
    torch.set_num_threads(1)
    args = SimpleNamespace(seed=6, payload=24, depth=2, heads=2, pool=4,
        input_init='balanced', tie_pools=True, readout='adaptive', readout_init='frequency',
        minimum_tail_width=0, state_mode='carry', route_credit='none', future_site='uniform',
        free_bias=0., temperature=1., compiled=False, credit_window=16)
    counts = torch.ones(50257, dtype=torch.long)
    order = torch.arange(50257)
    ids = torch.arange(100, 116).reshape(1, 16)
    def build(scale=None):
        torch.manual_seed(6)
        model = (PrivateBankModel(args, order, counts) if scale is None else
                 TimeScaledPrivateBankModel(args, order, counts, scale))
        # Frequency initialization has zero output weights: first-step core
        # gradients are exactly zero. Warm only the decoder, identically for
        # every arm, to test temporal learning after that known cold start.
        x, _, _ = model(ids, None, torch.Generator().manual_seed(9), args)
        model.readout.nll(x.detach(), ids + 1).mean().backward()
        torch.optim.SGD(model.readout.parameters(), lr=.01).step()
        model.zero_grad(set_to_none=True)
        return model
    def factual(model):
        x, state, _ = model(ids, None, torch.Generator().manual_seed(9), args)
        loss = model.readout.nll(x, ids + 1).mean()
        loss.backward()
        return x.detach(), state, loss.detach()
    base = build()
    xb, _, lb = factual(base)
    rows = []
    for scale in (1., .25, .0625):
        model = build(scale)
        x, _, loss = factual(model)
        assert torch.isfinite(loss)
        for value in model.parameters():
            if value.grad is not None: assert torch.isfinite(value.grad).all()
        if scale == 1:
            assert torch.equal(xb, x) and torch.equal(lb, loss)
            for name, value in base.named_parameters():
                other = dict(model.named_parameters())[name]
                assert torch.equal(value, other), name
                assert (value.grad is None) == (other.grad is None), name
                if value.grad is not None: assert torch.equal(value.grad, other.grad), name
        norms = {name: float(model.core.banks[name].grad.norm()) for name in ('raw_rate', 'frequency')}
        assert all(value > 0 for value in norms.values()), (scale, norms)
        model.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(19)
        before = generator.get_state()
        x, state, pis = model(ids, None, generator, args)
        pi = pis[2 * args.depth + 1][1][:, 0, :].double()
        winner = state['race_winners'][2, 1, :, 0]
        alternative = (winner + 1) % args.pool
        # Deterministic proposal contract: q=1 for the designated alternative.
        model.force_site = (2, 1, 0, alternative)
        shadow_gen = torch.Generator(); shadow_gen.set_state(before)
        with torch.no_grad():
            shadow, _, _ = model(ids, None, shadow_gen, args)
            difference = (model.readout.nll(shadow[:, 2:], ids[:, 2:] + 1) -
                          model.readout.nll(x[:, 2:], ids[:, 2:] + 1)).reshape(ids.shape[0], -1).mean(1)
        model.force_site = None
        term = paired_route_credit(pi, alternative, torch.ones_like(difference), difference)
        assert float(term.detach()) == 0 and torch.isfinite(difference).all()
        term.backward()
        credit_norm = float(model.core.banks['key'].grad.norm())
        assert credit_norm > 0 and torch.isfinite(model.core.banks['key'].grad).all()
        # Persistent numerical state plus route RNG must continue exactly.
        payload = io.BytesIO()
        torch.save(dict(model=model.state_dict(), state=state, rng=generator.get_state()), payload)
        payload.seek(0); saved = torch.load(payload, weights_only=True)
        restored = build(scale); restored.load_state_dict(saved['model'])
        g2 = torch.Generator(); g2.set_state(saved['rng'])
        with torch.no_grad():
            expected, _, _ = model(ids[:, :4], state, generator, args)
            actual, _, _ = restored(ids[:, :4], saved['state'], g2, args)
        assert torch.equal(expected, actual) and torch.equal(generator.get_state(), g2.get_state())
        wrong = build(.5 if scale != .5 else .25)
        try: wrong.load_state_dict(saved['model'])
        except ValueError: pass
        else: raise AssertionError('Cross-scale checkpoint accepted')
        rows.append(dict(scale=scale, factual_nll=float(loss), temporal_gradient_norms=norms,
                         alternative_suffix_difference=float(difference.mean()), route_credit_norm=credit_norm))
    check()
    record = dict(status='completed', rows=rows, source_sha256=pins,
        scale_one_factual_gradient_exact=True, scaled_temporal_gradients_nonzero=True,
        actual_alternative_suffix_credit_nonzero=True, model_state_rng_continuation_exact=True,
        cross_scale_checkpoint_rejected=True, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Synthetic integrated token contracts; deterministic alternative proposal, not sampled estimator validation. Model/state/RNG continuation only; optimizer-resume and trained quality require separate evidence.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
