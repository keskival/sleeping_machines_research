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


def nll(z, y, g, mask, parts=False):
    """per-position NLL (type + log-gap Gaussian); z: (n, T, 48).  parts: return (type, gap) separately."""
    ce = F.cross_entropy(z[..., :V].reshape(-1, V), y.reshape(-1), reduction='none').view(y.shape)
    mu = z[..., V].double(); sigma = F.softplus(z[..., V + 1]).double() + 1e-3
    gauss = .5 * ((g - mu) / sigma) ** 2 + sigma.log() + .5 * math.log(2 * math.pi)
    if parts:
        return ce.double() * mask, gauss * mask
    return (ce.double() + gauss) * mask


RULES = ('total', 'type', 'gap', 'gap_window32_max')   # declared 5 Oct 00:10 UTC before any further test scoring;
WINDOW = 32                                           # 'total' (mean NLL over the prefix) stays the primary rule


def prefix_scores(per_type, per_gap, lengths):
    """rule -> (n_runs, len(PREFIXES)) scores for prefixes of N events (predictions 1..N-1); NaN if the run is shorter."""
    out = {r: np.full((len(lengths), len(PREFIXES)), np.nan) for r in RULES}
    ct, cg = np.cumsum(per_type, 1), np.cumsum(per_gap, 1)
    for r, n_ev in enumerate(lengths):
        for k, N in enumerate(PREFIXES):
            if N > n_ev:
                continue
            m = N - 1
            out['type'][r, k] = ct[r, m - 1] / m; out['gap'][r, k] = cg[r, m - 1] / m
            out['total'][r, k] = out['type'][r, k] + out['gap'][r, k]
            wl = min(WINDOW, m); c = np.concatenate([[0.], cg[r, :m]])
            out['gap_window32_max'][r, k] = ((c[wl:] - c[:-wl]) / wl).max()
    return out


def scores(model, runs, lanes, fn, recruit_kw=None):
    """per-rule prefix scores for every run, the mean per-event NLL, and slot occupancy (recruit_kw path only).
    recruit_kw: evaluate through carried_logits with the recruitment layer (neutral settings equal the standard
    layer; THEORY §419), padding each batch to its longest run by repeating the last event (padded positions are
    never scored); occupancy is read from the written flags after the batch's shortest run."""
    parts = []; total = 0.; count = 0; used = []; ever = None
    model.eval()
    with torch.no_grad():
        for b in range(0, len(runs), lanes):
            idx = list(range(b, min(b + lanes, len(runs))))
            rows, y, g, mask = batch(runs, idx)
            if recruit_kw is None:
                z = fn(model, rows, 314159, all_logits=True)
            else:
                from sleeping_machines.carried_episodes import carried_logits
                T = y.shape[1]; lens = [len(runs[j][0]) for j in idx]; L0 = min(lens)
                st_np = np.stack([np.concatenate([runs[j][1], np.full(T - len(runs[j][1]), runs[j][1][-1])]) for j in idx])
                id_np = np.stack([np.concatenate([runs[j][0], np.full(T - len(runs[j][0]), runs[j][0][-1])]) for j in idx])
                stamps = torch.from_numpy(st_np).to(torch.float64); marks = torch.from_numpy(EYE[id_np])
                z1, st, _ = carried_logits(model, stamps[:, :L0], marks[:, :L0], seed=314159, route_credit=None,
                                           recruit=recruit_kw)
                seen = torch.stack(st['seen'])                                  # (D, n, H, U)
                used.append(seen.sum(-1).float().mean(1).numpy())               # (D, H): slots written per run
                ever = seen.any(1) if ever is None else ever | seen.any(1)      # (D, H, U)
                if T > L0:
                    z2, _, _ = carried_logits(model, stamps[:, L0:], marks[:, L0:], state=st, seed=314160,
                                              route_credit=None, recruit=recruit_kw)
                    z = torch.cat([z1, z2], 1)
                else:
                    z = z1
            pt, pg = nll(z, y, g, mask, parts=True)
            pt, pg = pt.numpy(), pg.numpy(); total += pt.sum() + pg.sum(); count += int(mask.sum())
            parts.append(prefix_scores(pt, pg, [len(runs[j][0]) for j in idx]))
    occ = None
    if used:
        U = ever.shape[-1]
        occ = dict(slots_written_per_run=np.mean(used, 0).round(3).tolist(), pool=U,
                   dead_slot_fraction=(1 - ever.float().mean(-1)).numpy().round(3).tolist(),
                   scope='per layer x head; written flags after each batch\'s shortest run')
    return {r: np.concatenate([q[r] for q in parts]) for r in RULES}, total / max(count, 1), occ


def aurocs(sc_clean, sc_faulty, kinds):
    """sc_*: rule -> scores (or one array, treated as the primary rule).  Returns the primary rule's table, with the
    other rules under 'rules'."""
    if not isinstance(sc_clean, dict):
        sc_clean, sc_faulty = {'total': sc_clean}, {'total': sc_faulty}
    res = {}
    for rule in sc_clean:
        tab = {}
        for k, N in enumerate(PREFIXES):
            a, b = sc_clean[rule][:, k], sc_faulty[rule][:, k]
            a, keep = a[~np.isnan(a)], ~np.isnan(b)
            if len(a) == 0 or keep.sum() == 0:
                continue
            row = dict(all=auroc(a, b[keep]))
            for f, name in ((1, 'wear_and_tear'), (2, 'retry_delay')):
                row[name] = auroc(a, b[keep & (kinds == f)]) if (keep & (kinds == f)).any() else None
            tab[N] = row
        res[rule] = tab
    out = dict(res['total']); out['rules'] = {r: v for r, v in res.items() if r != 'total'}
    return out


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
    p.add_argument('--tau-max', type=float, default=0., help='initialize unit memory time constants log-spaced from 1 to '
                   'this many seconds (default 0: unit default 1-100 s); R0 learned median half-lives ~7 s while one item '
                   'spends ~720 s on the line')
    p.add_argument('--findable-init', action='store_true', help='§419 init: key_read = 0 (free and occupied slots start on '
                   'equal key-only footing) and orthogonal per-head keys of norm --key-norm')
    p.add_argument('--key-norm', type=float, default=1.)
    p.add_argument('--tie-pools', action='store_true', help='share each pool\'s maps (§398); removes the untrained-loser bias (§419)')
    p.add_argument('--free-bias', type=float, default=0., help='fixed score bonus for never-written slots (§419; training and evaluation)')
    p.add_argument('--stale-bias', type=float, default=0., help='score bonus x (1 - retained fraction) of each slot; never-written = 1 (§419 graded optionality)')
    p.add_argument('--train-temperature', type=float, default=1., help='race score temperature in training only (§419 exposure)')
    p.add_argument('--balance', type=float, default=0., help='load-balance (importance) penalty weight on race probabilities (§419)')
    p.add_argument('--segment', type=int, default=0, help='truncated credit: segments of this many events with the '
                   'state carried (detached) across segments (sleeping_machines/carried_episodes.py); 0: whole runs')
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
    if a.tau_max:
        with torch.no_grad():
            for layer in model.units:
                for head in layer:
                    for pool in head:
                        for unit in pool:
                            tau = torch.logspace(0, math.log10(a.tau_max), unit.raw_rate.numel(), dtype=unit.raw_rate.dtype)
                            unit.raw_rate.copy_(torch.expm1(1 / tau).log())
    if a.findable_init:
        with torch.no_grad():
            for layer in model.units:
                for head in layer:
                    units = [u for pool in head for u in pool]
                    Q = torch.linalg.qr(torch.randn(a.payload, max(a.payload, len(units))))[0]
                    for k, unit in enumerate(units):
                        unit.key_read.weight.zero_()
                        unit.key.copy_(Q[:, k % Q.shape[1]] * a.key_norm)
    if a.tie_pools:
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    recruit = (a.free_bias != 0. or a.stale_bias != 0. or a.train_temperature != 1. or a.balance > 0.)
    rkw = dict(free_bias=a.free_bias, stale_bias=a.stale_bias, temperature=1., eager=not a.compiled) if a.segment else None   # evaluation at tau = 1
    fn = batched_logits
    if a.compiled:
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config
        from sleeping_machines.compiled_episodes import compiled_logits
        inductor_config.compile_threads = 1; dynamo_config.cache_size_limit = 64
        fn = compiled_logits
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    per_epoch = math.ceil(len(train) / a.lanes) * (math.ceil(len(train[0][0]) / a.segment) if a.segment else 1)
    total = a.epochs * per_epoch
    if a.segment and len({len(r[0]) for r in train}) != 1:
        raise ValueError('--segment needs equal-length training runs: set --train-max-events <= the shortest run')
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
            if a.segment:                   # truncated credit: consecutive segments, state carried (detached)
                from sleeping_machines.carried_episodes import carried_logits, detach
                from sleeping_machines.compiled_episodes import layer_step
                stamps = torch.tensor(np.stack([train[j][1] for j in idx]), dtype=torch.float64)
                marks = torch.from_numpy(np.stack([EYE[train[j][0]] for j in idx]))
                state = None
                for s in range(0, stamps.shape[1], a.segment):
                    if w >= total:
                        break
                    sl = slice(s, s + a.segment)

                    def seg_step(step_fn):
                        nonlocal state
                        model.train(); opt.zero_grad(set_to_none=True)
                        if recruit:
                            z, st, pis = carried_logits(model, stamps[:, sl], marks[:, sl], state=state, seed=100000 + w,
                                                        route_credit=rc, recruit=dict(free_bias=a.free_bias, stale_bias=a.stale_bias,
                                                        temperature=a.train_temperature, eager=not a.compiled))
                        else:
                            z, st = carried_logits(model, stamps[:, sl], marks[:, sl], state=state, seed=100000 + w,
                                                   step=step_fn, route_credit=rc)
                        per = nll(z, y[:, sl], g[:, sl], mask[:, sl]); m = mask[:, sl].sum()
                        loss = per.sum() / m
                        if recruit and a.balance > 0:
                            from sleeping_machines.recruit_layer import balance_penalty
                            loss = loss + a.balance * balance_penalty(pis)
                        loss.float().backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step()
                        state = detach(st); box['loss'] = float(loss.detach()); box['n'] = int(m)
                    if w < a.trace_windows:     # work trace: the batched formulation (same per-event operators), fresh state
                        seg_rows = [dict(events=[(float(stamps[r, k]), marks[r, k].numpy()) for k in range(sl.start, min(sl.stop, stamps.shape[1]))])
                                    for r in range(stamps.shape[0])]

                        def traced():
                            model.train(); opt.zero_grad(set_to_none=True)
                            per = nll(batched_logits(model, seg_rows, 100000 + w, all_logits=True, route_credit=rc),
                                      y[:, sl], g[:, sl], mask[:, sl]); m = mask[:, sl].sum()
                            loss = per.sum() / m
                            loss.float().backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step()
                            box['loss'] = float(loss.detach()); box['n'] = int(m)
                        rec = capture(traced); ledger = merge([ledger, rec]) if ledger else rec
                        traced_events += stamps.shape[0] * (min(s + a.segment, stamps.shape[1]) - s)
                    else:
                        seg_step(None if a.compiled else layer_step)
                    schedule.step(); w += 1
                    events_seen += stamps.shape[0] * (min(s + a.segment, stamps.shape[1]) - s)
                    loss_sum += box['loss'] * box['n']; n_sum += box['n']
            else:
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
        sc_c, val_nll, occ = scores(model, val_c, 64, fn, rkw); sc_f, _, _ = scores(model, val_f, 64, fn, rkw)
        curve.append(dict(epoch=epoch, train_nll=loss_sum / max(n_sum, 1), val_clean_nll=val_nll,
                          val_auroc=aurocs(sc_c, sc_f, val_k), val_clean_occupancy=occ, epoch_s=time.perf_counter() - t0))
        print(json.dumps(curve[-1]), flush=True)
        if val_nll < best[0]:
            best = (val_nll, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if w >= total:
            break
    model.load_state_dict(best[2])
    weights = ROOT / 'experiments/results/fas/checkpoints' / f'{a.tag}_selected.pt'
    with torch.no_grad():                # §419 prediction: successful recruitment lengthens learned memory horizons
        half_lives = {}
        for depth, Lp in enumerate(model._stacked(0)):
            hl = (math.log(2) / Lp['rate'].double().flatten()).numpy()
            half_lives[depth] = dict(p10=float(np.quantile(hl, .1)), median=float(np.median(hl)), p90=float(np.quantile(hl, .9)),
                                     max=float(hl.max()), unit='seconds of simulator time, at unit forget gate (base rate only)')
    weights.parent.mkdir(exist_ok=True); torch.save(model.state_dict(), weights)
    test_c, _ = load(d / 'test_clean.npz', a.max_events); test_f, test_k = load(d / 'test_faulty.npz', a.max_events)
    if a.max_windows:                   # smoke: a bounded test subset, labelled by status
        test_c, test_f, test_k = test_c[:a.eval_runs], test_f[:a.eval_runs], test_k[:a.eval_runs]
    sc_c, test_nll, test_occ = scores(model, test_c, 64, fn, rkw); sc_f, _, _ = scores(model, test_f, 64, fn, rkw)
    work = (ledger['arithmetic_flops'] + ledger['special_function_evaluations']) / traced_events if traced_events else None
    result = dict(status='smoke' if a.max_windows else 'completed', args=vars(a),
                  parameters=sum(q.numel() for q in model.parameters()), curve=curve, selected_epoch=best[1], selection='validation-clean NLL only',
                  test_clean_nll=test_nll, test_clean_occupancy=test_occ, memory_half_lives=half_lives, test_auroc=aurocs(sc_c, sc_f, test_k), selected_weights=str(weights.relative_to(ROOT)),
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
