"""Small integrated fit: does posterior-routed binding emerge from scratch? (battle B3; THEORY §§434, 436)

Synthetic interleaved processes with known identities:
- K processes per sample; each walks a fixed route of R event types (types 1..R) once;
- step r takes a log-normal duration with median m_r (distinct per step, CV --cv), and starts are staggered;
- each event is dropped with probability --drop;
- the merged log is anonymous.

A small binding-memory race-readout model (sleeping_machines/race_readout.py) trains on clean samples by the
filtering objective. The script reports, before and after training:
- validation NLL per event;
- binding purity alpha (consecutive writes to a slot from the same process).

Pass criterion for the C1 mechanism: alpha rises from chance (~1/K) toward 1 while validation NLL falls.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.race_readout import BindingMemory, PredictiveLayer, RaceReadout, detach_state, readout_episode  # noqa: E402


def sample(rng, K, R, medians, cv, stagger, drop):
    ev = []
    for k in range(K):
        t = k * stagger + rng.exponential(stagger / 4)
        for r in range(R):
            t += medians[r] * math.exp(cv * rng.standard_normal())
            if rng.random() >= drop:
                ev.append((t, r + 1, k))
    ev.sort()
    t, e, w = (np.array(x) for x in zip(*ev))
    return e.astype(np.int64), t, w


def batch(samples, T):
    st = np.stack([np.concatenate([s[1][:T], np.full(max(0, T - len(s[1])), s[1][-1])])[:T] for s in samples])
    ids = np.stack([np.concatenate([s[0][:T], np.full(max(0, T - len(s[0])), s[0][-1])])[:T] for s in samples])
    who = np.stack([np.concatenate([s[2][:T], np.full(max(0, T - len(s[2])), -1)])[:T] for s in samples])
    lens = [min(len(s[0]), T) for s in samples]
    return torch.from_numpy(st), torch.from_numpy(np.eye(V_TOY, dtype=np.float32)[ids]), torch.from_numpy(ids), who, lens


V_TOY = 0


def alpha_of(slots, who, lens):
    same = n = 0
    for i, L in enumerate(lens):
        sl, w = slots[i, :L], who[i, :L]
        for s in np.unique(sl):
            x = w[sl == s]; same += int((x[1:] == x[:-1]).sum()); n += len(x) - 1
    return same / max(n, 1)


SKIP = dict(deep=False, pred=None)


def evaluate(model, ro, bm, data, T, deterministic=False):
    stamps, marks, ids, who, lens = batch(data, T)
    rec = []
    with torch.no_grad():
        ll, _, valid, _ = readout_episode(model, ro, stamps, marks, ids, seed=1, binding=bm, record=rec,
                                          deterministic=deterministic, skip_deep=SKIP['deep'], pred_layers=SKIP['pred'])
    mask = torch.from_numpy(np.arange(T)[None] < np.asarray(lens)[:, None]) & valid
    slots = torch.stack([r['slot'] for r in rec], 1).numpy()
    return float(-(ll * mask).sum() / mask.sum()), alpha_of(slots, who, lens)


def main():
    global V_TOY
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--K', type=int, default=3); p.add_argument('--R', type=int, default=8)
    p.add_argument('--cv', type=float, default=.15); p.add_argument('--drop', type=float, default=0.)
    p.add_argument('--stagger', type=float, default=8.); p.add_argument('--train', type=int, default=256)
    p.add_argument('--val', type=int, default=64); p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lanes', type=int, default=16); p.add_argument('--slots', type=int, default=6)
    p.add_argument('--classes', type=int, default=2); p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--out', default='')
    p.add_argument('--gated', action='store_true')
    p.add_argument('--train-argmax', action='store_true', help='hard-EM writes in training (deterministic races)')
    p.add_argument('--hidden', type=int, default=32); p.add_argument('--no-context', action='store_true')
    p.add_argument('--additive-context', action='store_true')
    p.add_argument('--skip-deep', action='store_true', help='no deep race layers (THEORY §438 ablation)')
    p.add_argument('--pred-layer', type=int, default=0, help='a predictive-routed deep layer with this many slots in place '
                   'of the learned deep layers (THEORY §438); its local likelihood joins the loss')
    p.add_argument('--learned-routing', action='store_true', help='comparison: query/key write race with linear write credit (§437 T1)')
    p.add_argument('--particles', default='', help='after training: SMC evaluation of validation NLL with these particle '
                   'counts (THEORY §434.1.2), e.g. 1,4,16')
    a = p.parse_args()
    SKIP['deep'] = a.skip_deep
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    V_TOY = a.R + 1
    medians = np.exp(np.linspace(math.log(3.), math.log(20.), a.R))[rng.permutation(a.R)]
    data = [sample(rng, a.K, a.R, medians, a.cv, a.stagger, a.drop) for _ in range(a.train + a.val)]
    train, val = data[:a.train], data[a.train:]
    T = a.K * a.R
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=V_TOY, classes=V_TOY + 2, payload=8, depth=1, heads=2, pool=2)
    ro = RaceReadout(V_TOY, 1, a.slots, 8, 16, hidden=a.hidden, mu0=2., classes=a.classes,
                     context='additive' if a.additive_context else not a.no_context)
    bm = BindingMemory(16, a.slots, 8, tau_max=100., gated=a.gated, learned=a.learned_routing)
    if a.pred_layer:
        SKIP['deep'] = True
        SKIP['pred'] = torch.nn.ModuleList([PredictiveLayer(V_TOY, 16, a.pred_layer, 8, hidden=a.hidden, classes=a.classes,
                                                            tau_max=100.)])
    params = list(model.parameters()) + list(ro.parameters()) + list(bm.parameters()) + (list(SKIP['pred'].parameters()) if SKIP['pred'] else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    curve = [dict(step=0, val=evaluate(model, ro, bm, val, T))]
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(train), a.lanes)
        stamps, marks, ids, _, lens = batch([train[i] for i in idx], T)
        local = []
        ll, _, valid, _ = readout_episode(model, ro, stamps, marks, ids, seed=step, binding=bm,
                                          deterministic=a.train_argmax, skip_deep=SKIP['deep'], pred_layers=SKIP['pred'],
                                          local=local)
        mask = torch.from_numpy(np.arange(T)[None] < np.asarray(lens)[:, None]) & valid
        loss = -(ll * mask).sum() / mask.sum()
        if local:                      # layer-local superposition likelihoods (§438)
            inside = torch.from_numpy(np.arange(T)[None] < np.asarray(lens)[:, None])
            for k, (lll, lv) in enumerate(local):
                loss = loss - (lll * (lv & inside[:, k])).sum() / mask.sum()
        opt.zero_grad(); loss.float().backward(); torch.nn.utils.clip_grad_norm_(params, 1.); opt.step()
        if step % max(1, a.steps // 6) == 0 or step == a.steps:
            curve.append(dict(step=step, train_nll=float(loss), val=evaluate(model, ro, bm, val, T),
                              val_argmax=evaluate(model, ro, bm, val, T, deterministic=True)))
            print(json.dumps(curve[-1]), flush=True)
    smc = {}
    if a.particles:
        sys.path.insert(0, str(ROOT / 'experiments/fas')); sys.path.insert(0, str(ROOT / 'experiments'))
        from race_smc_eval import smc_log_z
        stamps, marks, ids, _, lens = batch(val, T)
        for L in map(int, a.particles.split(',')):
            lz = smc_log_z(model, ro, stamps, marks, ids, L, True, binding=bm).numpy()
            smc[L] = float(np.mean([-lz[i, n - 1] / (n - 1) for i, n in enumerate(lens)]))
            print(json.dumps(dict(particles=L, val_nll=smc[L])), flush=True)
    res = dict(args=vars(a), chance_alpha=1 / a.K, curve=curve, wall_s=time.perf_counter() - t0, smc_val_nll=smc,
               verdict='binding emerges' if curve[-1]['val'][1] > .8 else 'binding does not emerge at this budget')
    print(json.dumps(dict(start=curve[0], end=curve[-1], verdict=res['verdict'])))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
