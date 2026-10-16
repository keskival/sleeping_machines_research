#!/usr/bin/env python3
"""R1 GPU ladder analysis: scaling fits and scaling-failure diagnosis (experiments/runbooks/GPU_SCALING_LADDER.md).

Reads every completed <name>_* run and its diag log. Writes <name>_analysis.json, <name>_report.md and figures.
Fits per variant (development-slice loss dev_w1024; the public protocol only for runs scored once with --mode public):
- IsoFLOP: at each budget a parabola in log(parameters) -> compute-optimal size N*(C) and loss L*(C) (>= 3 arms).
- Frontier: lower envelope of all arms, fit L(C) = E + A C^-alpha (and local slopes between budgets).
- Depth probe: loss against depth at fixed width and budget.
- Variant comparison: frontier gap per budget and its trend (widening = better exponent).
- Published reference: modded-NanoGPT record curves (public protocol), training FLOPs from the reference shape
  (6N + 6 n_layer T d per token, the convention of curie_plan_b_training_flops_20261016_v1), only against public-scored arms.
Diagnostics per run (flags in the report):
- under-training: prequential loss still falling > 0.02 per eighth in the last eighth;
- dead depth: layers whose logit-lens gain is < 0.005 nats at the final diagnostic;
- residual blow-up: final-layer max |residual| / first-layer > 50, or sublayer update ratio > 1 deep in the stack;
- attention collapse: read heads with entropy < 0.3 nats; memory rates pinned at their initial range ends;
- optimisation: loss spikes, non-finite steps, clipped fraction > 0.5, update/weight ratio outliers per module;
- context use: development loss by history bucket (does 256-1024 improve faster than 0-16 with scale?);
- token roles: word-initial vs continuation loss trend with scale; slot-bonus gain trend with scale;
- execution: MFU and tokens/s by shape (implementation bottleneck, not an architecture measure).
"""
import argparse, json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
REF_FLOPS_PER_TOKEN = 800623104.0                                       # GPT-2 124M shape, 6N + 6 L T d (curie_plan_b_training_flops_20261016_v1)


def load_runs(out_dir, name):
    runs = []
    for p in sorted(out_dir.glob(f'{name}_*_s0.json')):
        r = json.loads(p.read_text())
        if r.get('status') != 'completed': continue
        diag = out_dir/f"{r['tag']}.diag.jsonl"; r['diag'] = [json.loads(l) for l in diag.open()] if diag.exists() else []
        pub = out_dir/f"{r['tag']}_public.json"; r['public'] = json.loads(pub.read_text())['public_w1024']['nll'] if pub.exists() else None
        runs.append(r)
    return runs


def fit_frontier(C, L):
    """L = E + A C^-alpha by grid over E (E < min L), log-linear least squares for A, alpha."""
    C, L = np.asarray(C, float), np.asarray(L, float)
    if len(C) < 3: return None
    best = None
    for E in np.linspace(0, L.min() - 1e-3, 400):
        y = np.log(L - E); X = np.vstack([np.ones_like(C), -np.log(C)]).T; coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = float(((X @ coef - y)**2).sum())
        if best is None or err < best[0]: best = (err, E, math.exp(coef[0]), coef[1])
    return dict(E=best[1], A=best[2], alpha=best[3], sse_log=best[0])


def isoflop(arms):
    if len(arms) < 3: return None
    x = np.log([a['parameters'] for a in arms]); y = np.array([a['dev_w1024'] for a in arms]); c = np.polyfit(x, y, 2)
    if c[0] <= 0: i = int(np.argmin(y)); return dict(N_opt=arms[i]['parameters'], L_opt=float(y[i]), fit='argmin (no convex fit)')
    xo = -c[1]/(2*c[0]); return dict(N_opt=float(math.exp(xo)), L_opt=float(np.polyval(c, xo)), fit='parabola in log N', interior=bool(x.min() <= xo <= x.max()))


def diagnose(r):
    flags, out = [], {}
    pe = r['prequential_by_eighth']
    if len(pe) >= 2 and pe[-2] - pe[-1] > 0.02: flags.append(f'under-trained: last-eighth drop {pe[-2] - pe[-1]:.3f}')
    diags = [d for d in r['diag'] if d.get('event') == 'diag']
    if diags:
        last = diags[-1]; lens = [l['logit_lens_nll'] for l in last['layers']]; gains = [lens[i - 1] - lens[i] for i in range(1, len(lens))]
        out['logit_lens_nll'] = lens; out['layer_gain'] = gains
        dead = [i + 1 for i, g in enumerate(gains) if g < 0.005]
        if dead: flags.append(f'dead depth: layers {dead} gain < 0.005 nats')
        mx = [l['resid_out_max_abs'] for l in last['layers']]; out['resid_max_abs'] = mx
        if mx[-1] > 50*max(mx[0], 1e-6): flags.append(f'residual growth x{mx[-1]/mx[0]:.0f} over depth')
        for i, l in enumerate(last['layers']):
            for k in ('tm_update_ratio', 'read_update_ratio', 'mlp_update_ratio'):
                if i >= len(last['layers'])//2 and l.get(k, 0) > 1.0: flags.append(f'layer {i} {k} {l[k]:.2f} > 1 (late layers rewriting the stream)')
            low = [h for h, e in enumerate(l.get('read_entropy_per_head', [])) if e < 0.3]
            if low: flags.append(f'layer {i} read heads {low} collapsed (entropy < 0.3)')
        out['bonus_gain'] = last.get('bonus_gain'); out['read_mean_distance'] = [l.get('read_mean_distance_per_head') for l in last['layers']]
        ur = {k: v['update_ratio'] for k, v in last['modules'].items()}; med = float(np.median(list(ur.values()))) if ur else 0
        outl = [k for k, v in ur.items() if med and (v > 10*med or v < med/10)]
        if outl: flags.append(f'update/weight ratio outliers: {outl}')
    if r['spikes']: flags.append(f"{len(r['spikes'])} loss spikes")
    if r['nonfinite']: flags.append(f"{len(r['nonfinite'])} non-finite steps")
    if r['clipped_frac'] > 0.5: flags.append(f"clipped {r['clipped_frac']:.2f} of steps")
    m = r['milestones'][-1]; out['history'] = m.get('history'); out['roles'] = {k: v['nll'] for k, v in m.get('roles', {}).items()}
    out['mfu'] = r.get('mean_mfu'); out['flags'] = flags
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--out-dir', default=str(ROOT/'experiments/results/gpu_ladder')); a = ap.parse_args()
    out_dir = Path(a.out_dir); runs = load_runs(out_dir, a.name)
    if not runs: print('no completed runs'); return
    for r in runs:
        r['budget'] = r['args'].get('budget_flops') or r['train_flops']; r['diagnosis'] = diagnose(r)
    plan_p = out_dir/f'{a.name}_plan.json'; kinds = {x['tag']: (x['kind'], x['budget']) for x in json.loads(plan_p.read_text())['arms']} if plan_p.exists() else {}
    for r in runs: r['kind'], r['budget'] = kinds.get(r['tag'], ('ladder', r['train_flops']))
    res = dict(name=a.name, runs=[{k: r[k] for k in ('tag', 'kind', 'variant', 'budget', 'parameters', 'train_flops', 'train_tokens', 'dev_w1024', 'select_w1024', 'flagship_w1024', 'public', 'mean_mfu', 'shape', 'diagnosis')}
                                  | dict(d=r['args']['d'], layers=r['args']['layers'], lr=r['lr']) for r in runs], variants={})
    for v in sorted({r['variant'] for r in runs}):
        vr = [r for r in runs if r['variant'] == v and r['kind'] in ('ladder', 'depth_probe', 'extra_variant', 'lr_sweep')]
        iso = {}
        for b in sorted({r['budget'] for r in vr}):
            arms = sorted([r for r in vr if r['budget'] == b and r['kind'] != 'lr_sweep'], key=lambda r: r['parameters'])
            f = isoflop(arms); iso[f'{b:.0e}'] = f if f else dict(arms=len(arms))
        pts = sorted((r['train_flops'], r['dev_w1024']) for r in vr); env, best = [], math.inf
        for c, l in pts:
            if l < best: best = l; env.append((c, l))
        opt = [(float(k), f['L_opt']) for k, f in iso.items() if f and 'L_opt' in f]
        res['variants'][v] = dict(isoflop=iso, envelope=env, frontier_fit=fit_frontier(*zip(*env)) if len(env) >= 3 else None,
                                  isoflop_frontier_fit=fit_frontier(*zip(*opt)) if len(opt) >= 3 else None,
                                  local_slopes=[dict(c0=env[i][0], c1=env[i + 1][0], dL_per_decade=(env[i][1] - env[i + 1][1])/math.log10(env[i + 1][0]/env[i][0])) for i in range(len(env) - 1)])
    spec = json.loads(plan_p.read_text())['spec'] if plan_p.exists() else {}; pr = spec.get('depth_probe')   # probe arms shared with the ladder carry the ladder kind
    dp = [r for r in runs if r['kind'] == 'depth_probe' or (pr and r['args']['d'] == pr['d'] and abs(math.log(r['budget']/pr['budget'])) < 1e-6 and r['args']['layers'] in pr['layers'])]
    res['depth_probe'] = {v: sorted([(r['args']['layers'], r['dev_w1024']) for r in dp if r['variant'] == v]) for v in sorted({r['variant'] for r in dp})}
    if 'flagship' in res['variants'] and len(res['variants']) > 1:
        res['variant_gaps'] = {}
        for v in res['variants']:
            if v == 'flagship': continue
            a_, b_ = res['variants']['flagship']['isoflop'], res['variants'][v]['isoflop']
            res['variant_gaps'][v] = {k: b_[k]['L_opt'] - a_[k]['L_opt'] for k in a_ if k in b_ and 'L_opt' in a_[k] and 'L_opt' in b_[k]}
    ref = json.loads((ROOT/'experiments/results/public_reference/published_record_curves_20261016_v1.json').read_text())
    res['published_reference'] = [dict(record=rc['record'], points=[dict(train_tokens=p['train_tokens'], train_flops_estimate=p['train_tokens']*REF_FLOPS_PER_TOKEN, public_nll=p['val_nll']) for p in rc['curve'] if p['train_tokens']])
                                  for rc in ref['records']]
    res['public_scored'] = [dict(tag=r['tag'], train_flops=r['train_flops'], public_nll=r['public'], dev_nll=r['dev_w1024']) for r in runs if r['public'] is not None]
    (out_dir/f'{a.name}_analysis.json').write_text(json.dumps(res, indent=1) + '\n')
    lines = [f'# {a.name}: scaling analysis', '', f'{len(runs)} completed runs. Loss = development slice (65,536 targets, 1024-token windows); FLOPs = measured training FLOPs.', '']
    for v, f in res['variants'].items():
        ff = f['isoflop_frontier_fit'] or f['frontier_fit']; lines += [f'## {v}', '', f"Frontier fit: {('E=%.3f A=%.3g alpha=%.4f' % (ff['E'], ff['A'], ff['alpha'])) if ff else 'needs 3 budgets'}", '',
                                                                       '| budget | arms | N* | L* |', '| --- | --- | --- | --- |']
        for k, i in f['isoflop'].items(): lines.append(f"| {k} | {i.get('arms', '')} | {i.get('N_opt', float('nan')):.3g} | {i.get('L_opt', float('nan')):.4f} |")
        lines.append('')
    if res.get('variant_gaps'): lines += ['## Variant gaps (L* variant - L* flagship)', '', json.dumps(res['variant_gaps']), '']
    if res['depth_probe']: lines += ['## Depth probe', '', json.dumps(res['depth_probe']), '']
    lines += ['## Diagnostics', '', '| run | dev | MFU | flags |', '| --- | --- | --- | --- |']
    for r in res['runs']: lines.append(f"| {r['tag']} | {r['dev_w1024']:.4f} | {(r['mean_mfu'] or 0):.2f} | {'; '.join(r['diagnosis']['flags']) or '-'} |")
    (out_dir/f'{a.name}_report.md').write_text('\n'.join(lines) + '\n')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for v in res['variants']:
            vr = [r for r in runs if r['variant'] == v]; ax[0].scatter([r['train_flops'] for r in vr], [r['dev_w1024'] for r in vr], s=12, label=v)
            env = res['variants'][v]['envelope']; ax[0].plot(*zip(*env), lw=1)
        for rc in res['published_reference']: ax[0].plot([p['train_flops_estimate'] for p in rc['points']], [p['public_nll'] for p in rc['points']], 'k--', lw=0.8, label=rc['record'][:28] + ' (public)')
        if res['public_scored']: ax[0].scatter([p['train_flops'] for p in res['public_scored']], [p['public_nll'] for p in res['public_scored']], marker='*', s=60, c='k', label='ours, public protocol')
        ax[0].set_xscale('log'); ax[0].set_xlabel('training FLOPs'); ax[0].set_ylabel('NLL'); ax[0].legend(fontsize=7)
        for r in runs:
            lens = r['diagnosis'].get('logit_lens_nll')
            if lens: ax[1].plot(range(1, len(lens) + 1), lens, lw=0.8, alpha=0.7)
        ax[1].set_xlabel('layer'); ax[1].set_ylabel('logit-lens NLL (diag batch)'); fig.tight_layout(); fig.savefig(out_dir/f'{a.name}_scaling.png', dpi=120)
    except Exception as error: print('figure skipped:', error)
    print(json.dumps({v: dict(fit=f['isoflop_frontier_fit'] or f['frontier_fit']) for v, f in res['variants'].items()}, indent=1))


if __name__ == '__main__':
    main()
