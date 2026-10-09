#!/usr/bin/env python3
"""Analysis for theory note 160 sects. 7 and 11 (per-event learning without BPTT; unshared feedback trained by the forward
updates). Reads the online1_* and online2_* Taxi DEV results and applies the criteria stated before the runs:
  sect. 7:  online_trace within 0.02 nats/event of BPTT (final DEV log-likelihood, mean over seeds) and clearly above
            online_local (every seed's online_trace above the mean online_local).
  sect. 11: online_kp feedback cosines rise toward 1; online_kp within 0.02 nats/event of online_trace (v2, same decay);
            online_fa below both.
Writes experiments/results/credit/<tag>.json. No training.
"""
import argparse, glob, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'experiments/results/credit'


def arm(prefix, name):
    rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(R / f'{prefix}_taxi_{name}_s*.json')))]
    if not rs:
        return None
    fin = np.array([r['final_dev_ll'] for r in rs]); best = np.array([r['best_dev_ll'] for r in rs])
    se = lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None
    out = dict(seeds=len(rs), final_dev_ll=float(fin.mean()), final_se=se(fin), best_dev_ll=float(best.mean()), best_se=se(best),
               per_seed_final=fin.tolist(), wall_s=float(np.mean([r['wall_s'] for r in rs])),
               updates=int(rs[0]['history'][-1]['updates']))
    cos = [r['history'][-1].get('feedback_cosine') for r in rs]
    if cos and cos[0]:
        out['final_feedback_cosine'] = {k: float(np.mean([c[k] for c in cos])) for k in cos[0]}
        out['feedback_cosine_by_epoch_seed0'] = [h.get('feedback_cosine') for h in rs[0]['history']]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    A = {f'{p}:{n}': arm(p, n) for p, n in [('online1', 'bptt'), ('online1', 'online_trace'), ('online1', 'online_local'),
                                             ('online2', 'online_trace'), ('online2', 'online_kp'), ('online2', 'online_fa')]}
    v = {}
    b, tr, lo = A['online1:bptt'], A['online1:online_trace'], A['online1:online_local']
    if b and tr and lo:
        gap = b['final_dev_ll'] - tr['final_dev_ll']
        v['sect7'] = dict(trace_minus_bptt=-gap, within_002=gap <= 0.02,
                          above_local=all(x > lo['final_dev_ll'] for x in tr['per_seed_final']),
                          trace_minus_local=tr['final_dev_ll'] - lo['final_dev_ll'])
        v['sect7']['pass'] = v['sect7']['within_002'] and v['sect7']['above_local']
    t2, kp, fa = A['online2:online_trace'], A['online2:online_kp'], A['online2:online_fa']
    if t2 and kp and fa:
        v['sect11'] = dict(kp_minus_trace=kp['final_dev_ll'] - t2['final_dev_ll'], fa_minus_trace=fa['final_dev_ll'] - t2['final_dev_ll'],
                           kp_minus_fa=kp['final_dev_ll'] - fa['final_dev_ll'],
                           kp_cosine=kp.get('final_feedback_cosine'), fa_cosine=fa.get('final_feedback_cosine'))
        v['sect11']['pass'] = (t2['final_dev_ll'] - kp['final_dev_ll'] <= 0.02 and fa['final_dev_ll'] < min(kp['final_dev_ll'], t2['final_dev_ll'])
                               and min(kp['final_feedback_cosine'].values()) > 0.9)
    res = dict(status='completed', tag=a.tag, battle='ENABLER (theory note 160 sects. 7, 11)', arms=A, verdicts=v,
               units='DEV log-likelihood per scored event (nats; higher is better), Taxi, EasyTPP split',
               source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (R / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(v, indent=1))


if __name__ == '__main__':
    main()
