"""Credit fidelity audit (THEORY §429): implemented route credit versus exact counterfactual credit, on a trained FAS model.

For sampled races r (event k, layer d, head h) of a few validation runs:
- Exact: every alternative u is forced at r in a shadow lane of `batched_logits` (all other race-noise draws are
  shared, so lanes differ only in that choice). F_u is the lane's summed NLL after event k. The choice credit is
  c_u = pi_u (F_u - R), with R = sum_j pi_j F_j and pi from the factual race scores.
- Implemented: c_hat_u = dL/ds_u at r in the factual lane, under the training route credit (timing plus linear value
  credit, §413), where L is the summed NLL over the run.
- Horizon split of F_u - R: next (the prediction right after event k), window (the rest of k's training segment) and
  beyond (after that segment; truncated training never sees it).
Outputs: correlation and sign agreement of c_hat with c, and the share of |F_u - R| in each horizon part.
Evaluation only.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
from native import V, batch, load, nll  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402

OUT = ROOT / 'experiments/results/diagnostics'


def build(args, weights):
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=V, classes=V + 2, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    if args.get('tie_pools'):
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    model.load_state_dict(torch.load(weights, weights_only=True))
    return model.eval()


def per_position_nll(model, rows, y, g, mask, seed, forces=None, route_credit=None, record=None, grad=False):
    with torch.set_grad_enabled(grad):
        z = batched_logits(model, rows, seed, forces=forces, record=record, all_logits=True, route_credit=route_credit)
        return nll(z, y, g, mask)                                                # (n, T)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True, help='completed FAS native result JSON with selected_weights')
    p.add_argument('--tag', required=True); p.add_argument('--runs', type=int, default=4)
    p.add_argument('--events', type=int, default=320); p.add_argument('--races', type=int, default=6)
    p.add_argument('--segment', type=int, default=128); p.add_argument('--seed', type=int, default=5)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--estimator', choices=('implemented', 'compare'), default='implemented',
                   help='compare: also score value-only and transported write credit (sleeping_machines/transported_credit.py; '
                        'THEORY §430) on the same races; c_hat is then from the carried path with the same noise')
    a = p.parse_args()
    out = OUT / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); rng = np.random.default_rng(a.seed)
    res_path = Path(a.result).resolve(); res = json.loads(res_path.read_text()); args = res['args']
    model = build(args, ROOT / res['selected_weights'])
    D, H, U = model.depth, model.heads, model.pool
    runs, _ = load(ROOT / 'experiments/data/fas' / args['data'] / 'val_clean.npz', a.events + 1)
    runs = runs[:a.runs]
    rows, y, g, mask = batch(runs, list(range(len(runs))))
    seed = 314159
    record = []
    per = per_position_nll(model, rows, y, g, mask, seed, route_credit=args.get('route_credit', 'linear'), record=record, grad=True)
    bases = []                                  # record holds per-head views; the full per-layer scores carry both credit paths
    for s in record:
        b = s._base if s._base is not None else s
        if not any(b is x for x in bases):
            b.retain_grad(); bases.append(b)
    per.sum().backward()
    T = y.shape[1]
    extra = {}
    if a.estimator == 'compare':                # same runs, same noise realization, carried-path estimators (§430)
        from sleeping_machines.transported_credit import transported_logits
        st = torch.tensor(np.stack([r[1] for r in runs]), dtype=torch.float64)
        mk = torch.from_numpy(np.stack([np.eye(V, dtype=np.float32)[r[0]] for r in runs]))
        for name, wc in (('value_only', False), ('transported', True)):
            z, _, rec = transported_logits(model, st, mk, seed=seed, write_credit=wc)
            for _, _, s in rec:
                s.retain_grad()
            nll(z, y, g, mask).sum().backward()
            extra[name] = {(k, d): s.grad.detach().double().numpy() for k, d, s in rec}
            model.zero_grad()
    entries = []
    for i in range(len(runs)):
        ks = rng.integers(32, T - 64, a.races)
        for k in ks:
            d, h = rng.integers(0, D), rng.integers(0, H)
            r = int((k * D + d) * H + h)
            base = record[r]._base if record[r]._base is not None else record[r]
            scores = record[r][i].detach().double()
            c_hat = (base.grad[i, h] if record[r]._base is not None else base.grad[i]).double().numpy()
            pi = torch.softmax(scores, -1).numpy()
            forced_rows = [rows[i]] * U; forces = [(r, u) for u in range(U)]
            Fp = []
            for b in range(0, U, a.chunk):
                Fp.append(per_position_nll(model, forced_rows[b:b + a.chunk], y[[i] * len(forces[b:b + a.chunk])],
                                           g[[i] * len(forces[b:b + a.chunk])], mask[[i] * len(forces[b:b + a.chunk])],
                                           seed, forces=forces[b:b + a.chunk]).detach().numpy())
            Fp = np.concatenate(Fp)                                              # (U, T)
            seg_end = (k // a.segment + 1) * a.segment
            parts = dict(next=Fp[:, k], window=Fp[:, k + 1:seg_end].sum(1), beyond=Fp[:, seg_end:].sum(1))
            F = parts['next'] + parts['window'] + parts['beyond']
            R = float((pi * F).sum())
            c = pi * (F - R)
            dev = {n: v - float((pi * v).sum()) for n, v in parts.items()}
            more = {f'c_hat_{n}': v[(int(k), int(d))][i, h].tolist() for n, v in extra.items()}
            entries.append(dict(run=i, event=int(k), layer=int(d), head=int(h), pi=pi.tolist(), F_minus_R=(F - R).tolist(),
                                c=c.tolist(), c_hat=c_hat.tolist(), **more,
                                abs_share={n: float(np.abs(v).sum() / max(np.abs(F - R).sum(), 1e-12)) for n, v in dev.items()}))
    c_all = np.concatenate([e['c'] for e in entries]); h_all = np.concatenate([e['c_hat'] for e in entries])
    keep = np.abs(c_all) > 1e-9
    corr = float(np.corrcoef(c_all[keep], h_all[keep])[0, 1]) if keep.sum() > 2 else float('nan')
    sign = float((np.sign(c_all[keep]) == np.sign(h_all[keep])).mean()) if keep.any() else float('nan')
    shares = {n: float(np.mean([e['abs_share'][n] for e in entries])) for n in ('next', 'window', 'beyond')}
    by_estimator = {}
    for name in ['c_hat'] + [f'c_hat_{n}' for n in extra]:
        hh = np.concatenate([e[name] for e in entries])
        by_estimator[name] = dict(corr=float(np.corrcoef(c_all[keep], hh[keep])[0, 1]) if keep.sum() > 2 else float('nan'),
                                  sign_agreement=float((np.sign(c_all[keep]) == np.sign(hh[keep])).mean()) if keep.any() else float('nan'))
    result = dict(status='completed', args=vars(a), source_result=str(res_path.relative_to(ROOT)), model_args=args,
                  races=len(entries), pairs=int(keep.sum()), corr_chat_c=corr, sign_agreement=sign,
                  mean_abs_share_of_F_minus_R=shares, by_estimator=by_estimator, entries=entries,
                  scope='validation-clean runs, first --events process events, forced shadow lanes with shared noise; '
                        'c_hat = training score gradient (timing + linear value credit) of the summed run NLL')
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('races', 'pairs', 'corr_chat_c', 'sign_agreement', 'mean_abs_share_of_F_minus_R', 'by_estimator')}))


if __name__ == '__main__':
    main()
