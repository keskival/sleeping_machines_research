#!/usr/bin/env python3
"""Pilot kit: event-native anomaly detection on a partner's event log (EIC TRL-6 path; THEORY §440).

Input: a CSV with a timestamp column, an event-type column, and optionally a stream/entity column (machine, line, host,
patient). Every stream, or every fixed time window when no stream column is given, becomes one event sequence.

Pipeline:
1. Parse and map event types to ids. The timestamp resolution (smallest positive gap) becomes the recording cell, so
   clocks may not resolve below it (THEORY §439).
2. Train the race-of-delayed-clocks marked point process (the frozen B1 model, experiments/tpp/race_tpp_v5.py; public
   EasyTPP wins) on the reference period, assumed normal: windows ending before --reference-until, split 80/20
   train/dev by time.
3. Score every window with two statistics:
   - mean_nll: mean negative log-likelihood per event;
   - glr_max: the per-event-type slowdown GLR (THEORY §440.3). For each event type with >= 5 events, the best
     log-likelihood gain from scaling that type's gaps by e^s over s in a grid; maximum over types.
4. Calibrate alarm thresholds on the held-out reference (dev) windows to a 1% false-alarm rate per statistic.

Outputs in --out: scores.csv (one row per window, both statistics, alarm flags, driving event type), report.json
(model, data summary, thresholds, alarm counts), model.pt and vocabulary.json.
"""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v5 import RaceTPP, batches  # noqa: E402

GRID = [0., .05, .1, .2, .3, .5, .8]


def parse_time(v):
    try:
        return float(v)
    except ValueError:
        return dt.datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()


def read_log(path, time_col, event_col, stream_col):
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            rows.append((r[stream_col] if stream_col else '', parse_time(r[time_col]), r[event_col]))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def make_windows(rows, window_s, vocab):
    """sequences per stream; when window_s > 0 each stream is cut into consecutive windows of that length.
    Returns a list of (stream, start, end, times (rel. to start), marks)."""
    out = []
    by_stream = {}
    for s, t, e in rows:
        by_stream.setdefault(s, []).append((t, vocab.setdefault(e, len(vocab))))
    for s, ev in by_stream.items():
        ts = np.array([t for t, _ in ev]); ms = np.array([m for _, m in ev])
        if window_s > 0:
            edges = np.arange(ts[0], ts[-1] + window_s, window_s)
            for a in edges:
                k = (ts >= a) & (ts < a + window_s)
                if k.sum() >= 3:
                    out.append((s, a, a + window_s, ts[k] - a, ms[k]))
        elif len(ts) >= 3:
            out.append((s, ts[0], ts[-1], ts - ts[0], ms))
    return out


def per_event_ll(model, t, m, mask, scale=0.):
    """per-event log-likelihood (events 2..N) under gap scaling e^scale for every event; (B, L-1)."""
    h, slots = model.encode(t, m, mask)
    params = model.clocks(h[:, :-1], slots[:, :-1])
    tau = (t[:, 1:] - t[:, :-1]).clamp_min(0) * math.exp(-scale)
    log_h, log_s = model.clock_terms(tau, *params[:4])
    log_pk = params[4]
    log_lam_k = torch.logsumexp(log_h + log_pk.gather(-1, m[:, 1:, None, None].expand(-1, -1, model.M, 1)).squeeze(-1), -1)
    ll = log_lam_k + log_s.sum(-1) - scale
    return ll * mask[:, 1:], mask[:, 1:]


@torch.no_grad()
def score_windows(model, wins, batch=32):
    seqs = [(w[3], w[4]) for w in wins]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][1]))
    nll = np.zeros(len(seqs)); glr = np.zeros(len(seqs)); driver = [''] * len(seqs)
    for g0 in range(0, len(order), batch):
        g = order[g0:g0 + batch]
        n = max(len(seqs[i][1]) for i in g)
        t = np.zeros((len(g), n)); mk = np.zeros((len(g), n), np.int64); msk = np.zeros((len(g), n), bool)
        for r, i in enumerate(g):
            L = len(seqs[i][1]); t[r, :L] = seqs[i][0]; mk[r, :L] = seqs[i][1]; msk[r, :L] = True; t[r, L:] = seqs[i][0][-1]
        t, mk, msk = torch.from_numpy(t), torch.from_numpy(mk), torch.from_numpy(msk)
        lls = [per_event_ll(model, t, mk, msk, s)[0].numpy() for s in GRID]
        valid = msk[:, 1:].numpy(); marks = mk[:, 1:].numpy()
        for r, i in enumerate(g):
            v = valid[r]
            nll[i] = -lls[0][r][v].mean()
            best, who = 0., ''
            for e in np.unique(marks[r][v]):
                sel = v & (marks[r] == e)
                if sel.sum() >= 5:
                    gain = max((lls[j][r][sel] - lls[0][r][sel]).sum() for j in range(len(GRID))) / sel.sum()
                    if gain > best:
                        best, who = gain, int(e)
            glr[i] = best; driver[i] = who
    return nll, glr, driver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True); p.add_argument('--out', required=True)
    p.add_argument('--time-col', default='timestamp'); p.add_argument('--event-col', default='event')
    p.add_argument('--stream-col', default='')
    p.add_argument('--window-s', type=float, default=0., help='window length in seconds (0: one sequence per stream)')
    p.add_argument('--reference-until', required=True, help='timestamp; windows ending before it are the normal reference')
    p.add_argument('--epochs', type=int, default=60); p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--far', type=float, default=.01)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = random.Random(a.seed)
    started = time.time()
    rows = read_log(a.csv, a.time_col, a.event_col, a.stream_col)
    vocab = {}
    wins = make_windows(rows, a.window_s, vocab)
    cut = parse_time(a.reference_until)
    ref = sorted([w for w in wins if w[2] <= cut], key=lambda w: w[1]); mon = [w for w in wins if w[2] > cut]
    if len(ref) < 10:
        raise ValueError(f'only {len(ref)} reference windows; need at least 10')
    k = int(.8 * len(ref)); train, dev = ref[:k], ref[k:]
    gaps = np.concatenate([np.diff(w[3]) for w in train]); pos = gaps[gaps > 0]
    cell = float(pos.min()) if len(pos) else 0.
    scale = float(np.median(pos)); K = len(vocab)
    qs = np.log(np.quantile(pos, np.linspace(.1, .9, 8))).tolist()
    model = RaceTPP(K, 32, 16, 2, 2, 8, 4, .3, scale, qs, cell)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.)
    tr = [(w[3], w[4]) for w in train]; dv = [(w[3], w[4]) for w in dev]
    best, best_ep, state = -math.inf, -1, None
    for ep in range(a.epochs):
        model.train()
        for t, m, mask in batches(tr, 32, True, rng):
            tl, ml, n, _ = model.loglik(t, m, mask)
            opt.zero_grad(); (-(tl + ml) / n).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        model.eval()
        s = n = 0
        with torch.no_grad():
            for t, m, mask in batches(dv, 64, False, rng):
                tl, ml, nn_, _ = model.loglik(t, m, mask)
                s += (tl + ml).item(); n += int(nn_)
        if s / n > best:
            best, best_ep, state = s / n, ep, {k_: v.clone() for k_, v in model.state_dict().items()}
        if ep - best_ep >= a.patience:
            break
    model.load_state_dict(state); model.eval()
    d_nll, d_glr, _ = score_windows(model, dev)
    th = dict(mean_nll=float(np.quantile(d_nll, 1 - a.far)), glr_max=float(np.quantile(d_glr, 1 - a.far)))
    m_nll, m_glr, m_drv = score_windows(model, mon) if mon else (np.zeros(0), np.zeros(0), [])
    inv = {v: k_ for k_, v in vocab.items()}
    with open(out / 'scores.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['stream', 'start', 'end', 'events', 'mean_nll', 'glr_max', 'driving_event',
                                       'alarm_mean_nll', 'alarm_glr_max'])
        for i, win in enumerate(mon):
            w.writerow([win[0], win[1], win[2], len(win[4]), f'{m_nll[i]:.4f}', f'{m_glr[i]:.4f}',
                        inv.get(m_drv[i], '') if m_drv[i] != '' else '', int(m_nll[i] > th['mean_nll']),
                        int(m_glr[i] > th['glr_max'])])
    torch.save(model.state_dict(), out / 'model.pt'); (out / 'vocabulary.json').write_text(json.dumps(vocab, indent=1))
    report = dict(status='completed', csv=a.csv, event_types=K, windows=dict(train=len(train), dev=len(dev), monitored=len(mon)),
                  recording_cell=cell, best_dev_ll=best, best_epoch=best_ep, thresholds=th, target_false_alarm_rate=a.far,
                  alarms=dict(mean_nll=int((m_nll > th['mean_nll']).sum()), glr_max=int((m_glr > th['glr_max']).sum())),
                  parameters=sum(q.numel() for q in model.parameters()), wall_s=time.time() - started)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
