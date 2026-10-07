"""Error analysis for race-readout FAS models with binding memory (battle B3 development; THEORY §434.2, §435.3).
Evaluation only, validation split only. Identities from the v2 sidecar (identity.npz) are used for diagnosis, never for
fitting or scoring.

1. Binding purity (the alpha of Proposition 434.2): over consecutive writes to the same binding slot, the fraction
   whose true (line, item) is the same. Also item concentration: the share of each true item's events in its modal slot.
   Reported on clean and faulty validation runs.
2. Time-rescaling fit (Proposition 435.3): under the clean model the rescaled intervals dLambda_k are i.i.d. Exp(1).
   Same-millisecond events (ties) have zero elapsed time and dLambda = 0; their fraction is reported and the fit is
   measured on the positive intervals. Reports the Kolmogorov–Smirnov distance of u = 1 - exp(-dLambda) from uniform, the mean dLambda (1 under the model)
   and its quartiles on clean runs, plus the mean on faulty runs (faults add delay, so expect > 1).
3. Binding against detection: the total-rule AUROC at each prefix, with faulty runs split at the median purity
   (scored against all clean runs). Tests whether mis-binding is where detection is lost.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
from native import PREFIXES, V, auroc  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.race_readout import BindingMemory, RaceReadout, readout_episode  # noqa: E402

OUT = ROOT / 'experiments/results/fas'


def load_with_identity(d, split, max_events, runs):
    z = np.load(d / f'{split}.npz'); ident = np.load(d / 'identity.npz')
    o, ids, t = z['offsets'], z['ids'], z['times_ms']
    line, item = ident[f'{split}_line'], ident[f'{split}_item']
    out = []
    for r in range(min(runs, len(o) - 1)):
        sl = slice(o[r], o[r + 1]); m = ids[sl] != 0
        e = ids[sl][m][:max_events].astype(np.int64); tt = t[sl][m][:max_events] / 1000.
        who = (line[sl][m].astype(np.int64) * 1000 + item[sl][m])[:max_events]
        out.append((e, tt, who))
    return out, z['fault'][:runs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True); p.add_argument('--runs', type=int, default=200)
    p.add_argument('--lanes', type=int, default=32); p.add_argument('--tag', required=True)
    a = p.parse_args()
    out = Path(OUT) / f'{a.tag}.json'   # aws_benchmark redirects OUT as a string
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1)
    res = json.loads(Path(a.result).read_text()); args = res['args']
    if not args.get('binding_slots'):
        raise ValueError('binding-memory models only')
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=V, classes=V + 2, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    Ub = args['binding_slots']
    readout = RaceReadout(V, 1, Ub, args['payload'], args['heads'] * args['payload'], hidden=args['hidden'],
                          type_durations=args.get('type_durations', False), classes=args.get('step_classes', 0),
                              context=dict(full=True, additive='additive', none=False)[args.get('readout_context', 'full')])
    binding = BindingMemory(args['heads'] * args['payload'], Ub, args['payload'], tau_max=args.get('tau_max') or 1000.,
                                gated=args.get('binding_gated', False))
    wpath = Path(res['selected_weights']); wpath = wpath if wpath.is_absolute() else ROOT / wpath
    ck = torch.load(wpath, weights_only=True)
    model.load_state_dict(ck['model']); readout.load_state_dict(ck['readout']); binding.load_state_dict(ck['binding'])
    model.eval(); readout.eval(); binding.eval()
    d = ROOT / 'experiments/data/fas' / args['data']
    from native_race_readout import parts, prefix_scores, tensors
    report = dict(status='completed', battle='B3', source_result=a.result, runs=a.runs,
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/readout_diagnostics.py', 'sleeping_machines/race_readout.py')})
    per = {}
    for split in ('val_clean', 'val_faulty'):
        runs, kinds = load_with_identity(d, split, args['max_events'], a.runs)
        alpha, concentration, hazards, totals = [], [], [], []
        for b in range(0, len(runs), a.lanes):
            idx = list(range(b, min(b + a.lanes, len(runs))))
            stamps, marks, ids, lengths = tensors([(r[0], r[1]) for r in runs], idx)
            rec = []
            with torch.no_grad():
                ll, llt, _, _ = readout_episode(model, readout, stamps, marks, ids, seed=314159, binding=binding,
                                                record=rec, cell=args.get('cell_ms') / 1000. if args.get('cell_ms') else None)
            pt, pg, _ = parts(ll, llt, stamps, lengths)
            totals.append(prefix_scores(pt, pg, lengths)['total'])
            slots = torch.stack([r['slot'] for r in rec], 1).numpy()                       # (n, T)
            haz = [r['hazard'] for r in rec[1:]]
            hz = torch.stack(haz, 1).numpy() if haz else np.zeros((len(idx), 0))
            for i, j in enumerate(idx):
                L = lengths[i]; who = runs[j][2][:L]; sl = slots[i, :L]
                same = n_pairs = 0
                for s in np.unique(sl):
                    w = who[sl == s]; same += int((w[1:] == w[:-1]).sum()); n_pairs += len(w) - 1
                alpha.append(same / max(n_pairs, 1))
                conc = [np.bincount(sl[who == u]).max() / (who == u).sum() for u in np.unique(who)]
                concentration.append(float(np.mean(conc)))
                hazards.append(hz[i, :L - 1])
        h_all = np.concatenate(hazards)
        ties = float((h_all <= 1e-12).mean())                 # same-ms events: zero elapsed time, zero rescaled interval
        h = h_all[h_all > 1e-12]
        u = np.sort(1 - np.exp(-h)); n = len(u)
        ks = float(np.max(np.maximum(np.arange(1, n + 1) / n - u, u - np.arange(n) / n)))
        per[split] = dict(alpha=np.asarray(alpha), totals=np.concatenate(totals), kinds=np.asarray(kinds))
        report[split] = dict(binding_purity_alpha=dict(mean=float(np.mean(alpha)), median=float(np.median(alpha)),
                                                       p10=float(np.quantile(alpha, .1))),
                             item_concentration=float(np.mean(concentration)),
                             rescaled_interval=dict(mean=float(h.mean()), q25=float(np.quantile(h, .25)),
                                                    median=float(np.median(h)), q75=float(np.quantile(h, .75)),
                                                    ks_uniform=ks, n=int(n), zero_interval_fraction=ties,
                                                    scope='positive intervals only; ties at the clock resolution give 0',
                                                    exp1_reference=dict(mean=1., q25=.288, median=.693, q75=1.386)))
    c, f = per['val_clean'], per['val_faulty']
    hi = f['alpha'] >= np.median(f['alpha'])
    report['auroc_by_faulty_binding'] = {
        str(N): dict(all=auroc(c['totals'][:, j], f['totals'][:, j]),
                     high_purity=auroc(c['totals'][:, j], f['totals'][hi, j]),
                     low_purity=auroc(c['totals'][:, j], f['totals'][~hi, j]))
        for j, N in enumerate(PREFIXES)}
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'})[:2000])


if __name__ == '__main__':
    main()
