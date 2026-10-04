"""NeuroBench non-human-primate motor prediction (primate reaching) with the integrated native core
(experiments/SOTA_TARGETS.md, target 2).

Official data and preprocessing: neurobench 2.3.0 PrimateReaching (vendored, unmodified) with the leaderboard settings of
the baselines and the top entries (num_steps=1, train_ratio=0.5, bin_width=0.004, biological_delay=0,
remove_segments_inactive=False): 4 ms bins of multi-unit spikes (96 channels for indy, 192 for loco) and fingertip
velocity labels.  The test bins are dataset.ind_test (the last 25% of each of the official chunks); the first 75%
(ind_train + ind_val) is ours for fitting, as in the BioCAS 2024 top entries (fmi-basel/neural-decoding-RSNN), which
hold out the last part of it for validation.  R2 is the official NeuroBench R2: per output dimension
1 - SSE / (N var), averaged over x and y, over all test bins of a session; the leaderboard averages sessions.

Model: one event per 4 ms bin (timestamp = bin index, content = the bin's spike vector), the integrated native core with
route credit and a 2-output regression head (standardized velocity).  Training: random windows of the fitting bins,
MSE after a warmup; inference: the exact winner-only stepper over the test stream from a fresh state (as the top entry
evaluates its stream with batch size 1).  One model per session.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments/vendor/neurobench_2_3_0'))
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.sparse_inference import SparseStepper  # noqa: E402

DATA = ROOT / 'data/neurobench/primate_reaching'
SESSIONS = ['indy_20160622_01', 'indy_20160630_01', 'indy_20170131_02',
            'loco_20170210_03', 'loco_20170215_02', 'loco_20170301_05']


def load_session(name):
    from neurobench.datasets import PrimateReaching
    ds = PrimateReaching(file_path=str(DATA), filename=name, num_steps=1, train_ratio=0.5, bin_width=0.004,
                         biological_delay=0, remove_segments_inactive=False, download=False)
    spikes = ds.samples.T.numpy().astype(np.float32)        # (T, channels)
    labels = ds.labels.T.numpy().astype(np.float64)         # (T, 2) velocity
    fit = np.array(list(ds.ind_train) + list(ds.ind_val), dtype=np.int64)
    test = np.array(ds.ind_test, dtype=np.int64)
    return spikes, labels, fit, test


def r2(pred, label):
    """official NeuroBench R2 (metrics/workload/r2.py): mean over x, y of 1 - SSE / (N var)."""
    pred, label = np.asarray(pred, np.float64), np.asarray(label, np.float64)
    out = []
    for d in range(2):
        out.append(1 - np.sum((label[:, d] - pred[:, d]) ** 2) / (np.var(label[:, d]) * len(label)))
    return float(np.mean(out))


def make_model(a, channels):
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=channels, classes=2, payload=a.payload, depth=a.depth,
                                            heads=a.heads, pool=a.pool)
    if a.tie_pools:
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    return model


def with_traces(spikes, decays):
    """causal features: the bin's spikes and exponential traces tr_t = d tr_{t-1} + (1 - d) s_t along the recording's time
    axis, one per decay d (float16 storage)."""
    if not decays:
        return spikes
    T, C = spikes.shape
    out = np.empty((T, C * (1 + len(decays))), np.float16)
    out[:, :C] = spikes
    for j, d in enumerate(decays):
        tr = np.zeros(C, np.float32)
        col = out[:, C * (1 + j):C * (2 + j)]
        for t in range(T):
            tr = d * tr + (1 - d) * spikes[t]
            col[t] = tr
    return out


def run_session(a, name):
    spikes, labels, fit, test = load_session(name)
    spikes = with_traces(spikes, [float(d) for d in a.traces.split(',')] if a.traces else [])
    if a.val_blocks <= 1:
        n_val = int(round(len(fit) * a.val_fraction))
        train, val = fit[:len(fit) - n_val], fit[len(fit) - n_val:]
    else:                     # validation = the last val_fraction of each of val_blocks equal parts of the fitting bins
        parts = np.array_split(fit, a.val_blocks); train, val = [], []
        for part in parts:
            n_val = int(round(len(part) * a.val_fraction))
            train.append(part[:len(part) - n_val]); val.append(part[len(part) - n_val:])
        train, val = np.concatenate(train), np.concatenate(val)
    mu, sd = labels[train].mean(0), labels[train].std(0)
    target = ((labels - mu) / sd).astype(np.float32)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed + 1)
    model = make_model(a, spikes.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    S, B = a.segment, a.lanes
    rc = None if a.route_credit == 'none' else a.route_credit
    logits_fn = batched_logits
    if a.compiled:
        from sleeping_machines.compiled_episodes import compiled_logits
        logits_fn = compiled_logits
    losses = []
    started = time.perf_counter()
    for step in range(a.steps):
        starts = rng.integers(0, len(train) - S, B)
        idx = np.stack([train[s:s + S] for s in starts])                       # (B, S) bin indices
        rows = [dict(events=[(float(t), spikes[i].astype(np.float32)) for t, i in enumerate(r)]) for r in idx]
        y = torch.tensor(target[idx])
        model.train(); opt.zero_grad(set_to_none=True)
        out = logits_fn(model, rows, 1000 + step, all_logits=True, route_credit=rc)
        loss = F.mse_loss(out[:, a.warmup:], y[:, a.warmup:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); schedule.step()
        losses.append(float(loss))
        if step % 200 == 0:
            print(json.dumps(dict(session=name, step=step, loss=float(np.mean(losses[-50:])))), flush=True)
    fit_s = time.perf_counter() - started
    model.eval()

    def stream(indices, K=None):
        K = a.route_samples if K is None else K
        # K independent race-noise streams (separate steppers: one stepper shares each race's draw across its lanes);
        # predictions averaged
        steppers = [SparseStepper(model, 1, 777 + 1009 * k) for k in range(K)]; preds = []
        for t, i in enumerate(indices):
            x = torch.tensor(spikes[i].astype(np.float32))[None]
            preds.append(torch.stack([st.step(torch.tensor([float(t)]), x)[0] for st in steppers]).mean(0).numpy())
        return np.array(preds) * sd + mu
    single = None
    if a.route_samples > 1:                       # also the single-stream score, for the record
        single = dict(val_r2=r2(stream(val, 1), labels[val]) if len(val) else None, test_r2=r2(stream(test, 1), labels[test]))
    val_pred, test_pred = (stream(val) if len(val) else None), stream(test)
    val_r2 = r2(val_pred, labels[val]) if len(val) else None
    test_r2 = r2(test_pred, labels[test])
    smoothing = None
    if a.smoothing and len(val):          # causal leaky readout y_t = a y_{t-1} + (1-a) p_t, a chosen on validation only
        def leaky(pred, alpha):
            out = np.empty_like(pred); acc = pred[0]
            for t in range(len(pred)):
                acc = alpha * acc + (1 - alpha) * pred[t]; out[t] = acc
            return out
        grid = [0., .3, .5, .6, .7, .75, .8, .85, .9, .93, .95]
        scores = {alpha: r2(leaky(val_pred, alpha), labels[val]) for alpha in grid}
        best = max(scores, key=scores.get)
        smoothing = dict(alpha=best, val_r2=scores[best], grid={str(k): v for k, v in scores.items()},
                         test_r2=r2(leaky(test_pred, best), labels[test]))
    return dict(session=name, test_r2=test_r2 if smoothing is None else smoothing['test_r2'], unsmoothed_test_r2=test_r2,
                val_r2=val_r2, readout_smoothing=smoothing, route_samples=a.route_samples, single_stream=single, fit_bins=len(train), val_bins=len(val), test_bins=len(test),
                channels=int(spikes.shape[1]), final_train_mse=float(np.mean(losses[-50:])), fit_s=fit_s,
                parameters=sum(p.numel() for p in model.parameters()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--sessions', default=','.join(SESSIONS))
    p.add_argument('--payload', type=int, default=32); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--tie-pools', action='store_true')
    p.add_argument('--segment', type=int, default=250); p.add_argument('--lanes', type=int, default=32)
    p.add_argument('--steps', type=int, default=2000); p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--weight-decay', type=float, default=0.); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--warmup', type=int, default=25); p.add_argument('--val-fraction', type=float, default=.1333)
    p.add_argument('--route-credit', choices=('none', 'linear'), default='linear')
    p.add_argument('--compiled', action='store_true'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--route-samples', type=int, default=1, help='average predictions over this many race-noise streams '
                   '(inference work scales with it)')
    p.add_argument('--val-blocks', type=int, default=1, help='validation drawn from the end of this many equal parts of the '
                   'fitting bins (1: one block at the end)')
    p.add_argument('--traces', default='', help='comma list of per-bin decays for causal exponential spike traces added to '
                   'the content (e.g. .75,.95)')
    p.add_argument('--smoothing', action='store_true', help='causal leaky-integrator readout; constant chosen on validation')
    a = p.parse_args()
    out = ROOT / 'experiments/results/neurobench_primate' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); started = time.perf_counter()
    rows = []
    for name in a.sessions.split(','):
        rows.append(run_session(a, name)); print(json.dumps(rows[-1]), flush=True)
    result = dict(status='completed', args=vars(a), sessions=rows, mean_test_r2=float(np.mean([r['test_r2'] for r in rows])),
                  protocol='NeuroBench primate reaching (neurobench 2.3.0 loader, train_ratio .5, 4 ms bins); official R2',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/primate_reaching_native.py', 'sleeping_machines/sparse_inference.py',
                                  'sleeping_machines/batched_episodes.py', 'sleeping_machines/compiled_episodes.py',
                                  'experiments/vendor/neurobench_2_3_0/neurobench/datasets/primate_reaching.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mean_test_r2=result['mean_test_r2'])), flush=True)


if __name__ == '__main__':
    main()
