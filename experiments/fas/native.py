"""Native integrated core on the FAS interlaced-event anomaly benchmark (experiments/FAS_BENCHMARK.md).

Timestamped track.  Input events: the process (non-TICK) events of a run, timestamp = simulator time in seconds
(native computational time: memories decay and rotate with the elapsed time between events), content = one-hot event
id (46).  Training: clean runs only; next-event prediction = cross-entropy of the next event's type plus Gaussian NLL of
log(next gap + 10 ms) with predicted mean and scale (head outputs 46 type logits + 2).  Selection: validation-clean NLL
only.  Anomaly score of a run prefix of N process events: mean per-event NLL over the predictions inside the prefix.
AUROC (faulty positive) at each N on validation (development) and, with the selected weights, test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
from baselines import PREFIXES, auroc  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402

V = 46
EYE = np.eye(V, dtype=np.float32)
LOG_EPS = .01


def load(path, max_events):
    z = np.load(path); o, all_ids, all_t = z['offsets'], z['ids'], z['times_ms']; out = []   # decompress once
    for r in range(len(o) - 1):
        ids = all_ids[o[r]:o[r + 1]].astype(np.int64); t = all_t[o[r]:o[r + 1]] / 1000.
        m = ids != 0
        out.append((ids[m][:max_events], t[m][:max_events]))
    return out, z['fault']


def batch(runs, idx):
    rows = [dict(events=[(float(t), EYE[i]) for i, t in zip(*runs[j])]) for j in idx]
    T = max(len(runs[j][0]) for j in idx)
    y = torch.zeros(len(idx), T, dtype=torch.long); g = torch.zeros(len(idx), T, dtype=torch.float64)
    mask = torch.zeros(len(idx), T, dtype=torch.bool)
    for r, j in enumerate(idx):
        ids, t = runs[j]; n = len(ids) - 1
        y[r, :n] = torch.from_numpy(ids[1:]); g[r, :n] = torch.from_numpy(np.log(np.diff(t) + LOG_EPS)); mask[r, :n] = True
    return rows, y, g, mask


def nll(z, y, g, mask):
    """per-position NLL (type + log-gap Gaussian); z: (n, T, 48)."""
    ce = F.cross_entropy(z[..., :V].reshape(-1, V), y.reshape(-1), reduction='none').view(y.shape)
    mu = z[..., V].double(); sigma = F.softplus(z[..., V + 1]).double() + 1e-3
    gauss = .5 * ((g - mu) / sigma) ** 2 + sigma.log() + .5 * math.log(2 * math.pi)
    return (ce.double() + gauss) * mask


def scores(model, runs, lanes, fn):
    """mean NLL over the first N-1 predictions (prefix of N events) for every N in PREFIXES; NaN if shorter."""
    out = np.full((len(runs), len(PREFIXES)), np.nan); total = 0.; count = 0
    model.eval()
    with torch.no_grad():
        for b in range(0, len(runs), lanes):
            idx = list(range(b, min(b + lanes, len(runs))))
            rows, y, g, mask = batch(runs, idx)
            per = nll(fn(model, rows, 314159, all_logits=True), y, g, mask).numpy()
            total += per.sum(); count += int(mask.sum())
            csum = np.cumsum(per, 1)
            for r, j in enumerate(idx):
                n_ev = len(runs[j][0])
                for k, N in enumerate(PREFIXES):
                    if N <= n_ev:
                        out[j, k] = csum[r, N - 2] / (N - 1)
    return out, total / max(count, 1)


def aurocs(sc_clean, sc_faulty, kinds):
    res = {}
    for k, N in enumerate(PREFIXES):
        a, b = sc_clean[:, k], sc_faulty[:, k]
        a, keep = a[~np.isnan(a)], ~np.isnan(b)
        if len(a) == 0 or keep.sum() == 0:
            continue
        row = dict(all=auroc(a, b[keep]))
        for f, name in ((1, 'wear_and_tear'), (2, 'retry_delay')):
            row[name] = auroc(a, b[keep & (kinds == f)]) if (keep & (kinds == f)).any() else None
        res[N] = row
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--payload', type=int, default=32); p.add_argument('--depth', type=int, default=4)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--route-credit', default='linear'); p.add_argument('--compiled', action='store_true')
    p.add_argument('--epochs', type=int, default=3); p.add_argument('--lanes', type=int, default=8)
    p.add_argument('--fit-runs', type=int, default=10000); p.add_argument('--max-events', type=int, default=1100)
    p.add_argument('--train-max-events', type=int, default=0, help='train on the first K process events of each run '
                   '(0: --max-events); scoring always uses --max-events')
    p.add_argument('--lr', type=float, default=.003); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=6); p.add_argument('--eval-runs', type=int, default=1000)
    p.add_argument('--trace-windows', type=int, default=1); p.add_argument('--max-windows', type=int, default=0)
    a = p.parse_args()
    out = ROOT / 'experiments/results/fas' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); started = time.perf_counter()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = load(d / 'train_clean.npz', a.train_max_events or a.max_events); train = train[:a.fit_runs]
    val_c, _ = load(d / 'val_clean.npz', a.max_events); val_f, val_k = load(d / 'val_faulty.npz', a.max_events)
    val_c, val_f, val_k = val_c[:a.eval_runs], val_f[:a.eval_runs], val_k[:a.eval_runs]
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=V, classes=V + 2, payload=a.payload, depth=a.depth,
                                            heads=a.heads, pool=a.pool)
    fn = batched_logits
    if a.compiled:
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config
        from sleeping_machines.compiled_episodes import compiled_logits
        inductor_config.compile_threads = 1; dynamo_config.cache_size_limit = 64
        fn = compiled_logits
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    per_epoch = math.ceil(len(train) / a.lanes); total = a.epochs * per_epoch
    if a.max_windows:
        total = min(total, a.max_windows)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    rc = None if a.route_credit == 'none' else a.route_credit
    curve = []; best = (math.inf, None, None); ledger = None; traced_events = 0; events_seen = 0; w = 0
    from parallel_head_accumulated_language import merge
    from race_language_screen import capture
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(train)); t0 = time.perf_counter(); loss_sum = 0.; n_sum = 0
        for b in range(0, len(order), a.lanes):
            if w >= total:
                break
            idx = order[b:b + a.lanes]; rows, y, g, mask = batch(train, idx)
            box = {}

            def step(logits_fn):
                model.train(); opt.zero_grad(set_to_none=True)
                per = nll(logits_fn(model, rows, 100000 + w, all_logits=True, route_credit=rc), y, g, mask)
                loss = per.sum() / mask.sum()
                loss.float().backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step()
                box['loss'] = float(loss)
            if w < a.trace_windows:
                rec = capture(lambda: step(batched_logits)); ledger = merge([ledger, rec]) if ledger else rec
                traced_events += sum(len(r['events']) for r in rows)
            else:
                step(fn)
            schedule.step(); w += 1
            events_seen += sum(len(r['events']) for r in rows); loss_sum += box['loss'] * int(mask.sum()); n_sum += int(mask.sum())
            if w % 25 == 0:
                print(json.dumps(dict(window=w, of=total, train_nll=box['loss'],
                                      events_per_s=events_seen / (time.perf_counter() - started))), flush=True)
        sc_c, val_nll = scores(model, val_c, 64, fn); sc_f, _ = scores(model, val_f, 64, fn)
        curve.append(dict(epoch=epoch, train_nll=loss_sum / max(n_sum, 1), val_clean_nll=val_nll,
                          val_auroc=aurocs(sc_c, sc_f, val_k), epoch_s=time.perf_counter() - t0))
        print(json.dumps(curve[-1]), flush=True)
        if val_nll < best[0]:
            best = (val_nll, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if w >= total:
            break
    model.load_state_dict(best[2])
    test_c, _ = load(d / 'test_clean.npz', a.max_events); test_f, test_k = load(d / 'test_faulty.npz', a.max_events)
    if a.max_windows:                   # smoke: a bounded test subset, labelled by status
        test_c, test_f, test_k = test_c[:a.eval_runs], test_f[:a.eval_runs], test_k[:a.eval_runs]
    sc_c, test_nll = scores(model, test_c, 64, fn); sc_f, _ = scores(model, test_f, 64, fn)
    work = (ledger['arithmetic_flops'] + ledger['special_function_evaluations']) / traced_events if traced_events else None
    result = dict(status='smoke' if a.max_windows else 'completed', args=vars(a),
                  parameters=sum(q.numel() for q in model.parameters()), curve=curve, selected_epoch=best[1], selection='validation-clean NLL only',
                  test_clean_nll=test_nll, test_auroc=aurocs(sc_c, sc_f, test_k),
                  work=dict(fit_unit_special_flops_per_event_estimate=work,
                            whole_fit_unit_special_flops_estimate=work * events_seen if work else None,
                            fitting_events=events_seen, scope='first window traced (eager), extrapolated per event'),
                  data_manifest_sha256=hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest(),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/native.py', 'experiments/fas/baselines.py',
                                  'sleeping_machines/batched_episodes.py', 'sleeping_machines/compiled_episodes.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(test_auroc=result['test_auroc'], selected_epoch=best[1])), flush=True)


if __name__ == '__main__':
    main()
