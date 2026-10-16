"""R1 depth x data scaling (theory note 5, P5d; experiments/SCALING_SUPREMACY_PROTOCOL.md step 1).

Reads every completed one-pass plan-B run (r1_deep_keyed_stream.py; Curie and AWS tags) and reports, per depth and data
size: flagship (8192 targets, 1024-token windows), 65,528-target slice, selection slice and last-quarter prequential loss,
with measured training FLOPs (curie_plan_b_training_flops_20261016_v1). Decisions:
- P5d (registered 16 Oct): the 16M L2-L8 gap exceeds the 4M gap (flagship 0.023, last-quarter prequential 0.027).
- Data slope per depth: loss drop per 4x tokens (4M -> 16M); depth scales compositionally if deeper members drop faster.
- Compute frontier: for each budget, the depth with the lowest loss (does the compute-optimal depth grow with budget?).
"""
import argparse, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'experiments/results/token_language'
P5D_4M = dict(flagship=0.023, prequential_last_quarter=0.027)


def runs():
    out = []
    for p in sorted(RES.glob('*.json')):
        if not (p.name.startswith('curie_r1_deep_keyed_stream_') or p.name.startswith('aws_r1_depthdata_')): continue
        r = json.loads(p.read_text())
        if r.get('status') != 'completed' or 'stream' not in (r.get('stage', '') + str(r.get('source_sha256', {}))): continue
        g = r['args']; out.append(dict(tag=r['tag'], layers=g['layers'], d=g['d'], train_tokens=g['train_tokens'], parameters=r['parameters'],
                                       flagship=r['flagship_protocol']['nll'], score_65528=r['score_65528']['nll'], select=r['select']['nll'],
                                       prequential_last_quarter=r['prequential_last_quarter'], prequential_by_eighth=r['prequential_by_eighth'], train_wall_s=r['train_wall_s']))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    flops = {(m['d'], m['layers']): m['train_flops_per_target'] for m in json.loads((ROOT/'experiments/results/diagnostics/curie_plan_b_training_flops_20261016_v1.json').read_text())['members']}
    rs = runs()
    for r in rs:
        f = flops.get((r['d'], r['layers'])); r['train_flops'] = f*r['train_tokens'] if f else None
    by = {(r['d'], r['layers'], r['train_tokens']): r for r in rs}
    res = dict(battle='R1', tag=a.tag, runs=rs, decisions={})
    M4, M16 = 4194304, 16777216
    gaps = {}
    for size in (M4, M16):
        lo, hi = by.get((64, 2, size)), by.get((64, 8, size))
        if lo and hi: gaps[size] = {k: lo[k] - hi[k] for k in ('flagship', 'score_65528', 'select', 'prequential_last_quarter')}
    res['decisions']['L2_minus_L8_gap'] = {str(k): v for k, v in gaps.items()}
    if M16 in gaps:
        g = gaps[M16]; res['decisions']['P5d'] = dict(gap_16M=g, gap_4M_registered=P5D_4M,
                                                     passed=bool(g['flagship'] > P5D_4M['flagship'] and g['prequential_last_quarter'] > P5D_4M['prequential_last_quarter']))
    slopes = {}
    for L in (2, 4, 8, 16):
        a4, a16 = by.get((64, L, M4)), by.get((64, L, M16))
        if a4 and a16: slopes[L] = {k: a4[k] - a16[k] for k in ('flagship', 'score_65528', 'select')}
    res['decisions']['loss_drop_per_4x_tokens'] = slopes
    pts = sorted([r for r in rs if r['train_flops']], key=lambda r: r['train_flops']); frontier, best = [], math.inf
    for r in pts:
        if r['flagship'] < best: best = r['flagship']; frontier.append(dict(tag=r['tag'], layers=r['layers'], d=r['d'], train_tokens=r['train_tokens'], train_flops=r['train_flops'], flagship=r['flagship']))
    res['decisions']['flagship_compute_frontier'] = frontier; res['status'] = 'completed'
    path = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with path.open('x') as f: json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps(res['decisions'], indent=1))


if __name__ == '__main__':
    main()
