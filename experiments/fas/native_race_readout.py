"""FAS native model with the competing-risks race readout (THEORY note 155 §§432–433; battle B3 development).

The deep race network of native.py (AddressedEventHeads, carried segments) with the standard next-event head
replaced by sleeping_machines/race_readout.py. Each top-layer slot proposes its own next event (type law and own
duration since its last write). The merged next event is their race, and the likelihood is the exact superposition
density. --posterior also routes top-layer writes by the readout's posterior responsibility: the event is written where
it was predicted.

Scoring rules and prefixes are native.py's. Type and gap parts follow the same split: the gap part is the time-only
likelihood converted to log-gap units, log(gap + 0.01 s). The type part is the remainder, so per-event NLL is
comparable to native.py's.

Selection: validation-clean NLL. --no-test for development (B3 maturity gate). Development on fas_v1 validation is
disclosed and not counted in the v2 tuning budget.

Work: the first training window(s) run eagerly under the operation audit (work_audit.FasAudit; forward, backward and
optimizer) and are extrapolated per event. Inference work is traced on 4 validation runs.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
from native import EYE, LOG_EPS, PREFIXES, RULES, V, aurocs, load, prefix_scores  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.race_readout import BindingMemory, RaceReadout, detach_state, readout_episode  # noqa: E402


OUT = ROOT / 'experiments/results/fas'      # redirected by experiments/aws_benchmark.py


def _rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

def tensors(runs, idx):
    """pad to the longest run by repeating the last event; lengths returned."""
    T = max(len(runs[j][0]) for j in idx)
    st = np.stack([np.concatenate([runs[j][1], np.full(T - len(runs[j][1]), runs[j][1][-1])]) for j in idx])
    ids = np.stack([np.concatenate([runs[j][0], np.full(T - len(runs[j][0]), runs[j][0][-1])]) for j in idx])
    return (torch.from_numpy(st).to(torch.float64), torch.from_numpy(EYE[ids]), torch.from_numpy(ids).long(),
            [len(runs[j][0]) for j in idx])


def parts(ll_tot, ll_time, stamps, lengths):
    """per-position (type, gap) NLL in native.py's convention: position k predicts event k + 1."""
    lt, lti = ll_tot[:, 1:].numpy(), ll_time[:, 1:].numpy()
    gap = np.diff(stamps.numpy(), axis=1)
    pt = -(lt - lti); pg = -(lti + np.log(gap + LOG_EPS))
    mask = np.arange(lt.shape[1])[None] < (np.asarray(lengths)[:, None] - 1)
    return pt * mask, pg * mask, mask


GLR_GRID = [0., .05, .1, .2, .3, .5, .8]       # log slowdowns of the per-step GLR rule (THEORY §440.3)


def glr_prefix_scores(scaled, ids, lengths, min_events=5):
    """Per-step slowdown GLR: for prefix N, events 1..N-1, each event type e with >= min_events events gives
    G_e = max_s sum_k [ll_k(s) - ll_k(0)] / n_e over GLR_GRID; the score is max_e G_e. scaled: (n, T, G) per-event
    log-likelihoods under each slowdown (event 0 unused); ids (n, T). Returns (n, len(PREFIXES)), NaN if too short."""
    out = np.full((len(lengths), len(PREFIXES)), np.nan)
    gain = scaled - scaled[..., :1]
    for r, n_ev in enumerate(lengths):
        for j, N in enumerate(PREFIXES):
            if N > n_ev:
                continue
            g, e = gain[r, 1:N], ids[r, 1:N]
            best = -np.inf
            for t in np.unique(e):
                m = e == t
                if m.sum() >= min_events:
                    best = max(best, g[m].sum(0).max() / m.sum())
            out[r, j] = best if np.isfinite(best) else 0.
    return out


@torch.no_grad()
def scores(model, readout, runs, lanes, posterior, compiled, binding=None, cell=None, glr=False):
    out = []; total = 0.; count = 0; used = []
    model.eval(); readout.eval()
    for b in range(0, len(runs), lanes):
        idx = list(range(b, min(b + lanes, len(runs))))
        stamps, marks, ids, lengths = tensors(runs, idx)
        scaled = [] if glr else None
        ll, llt, _, st = readout_episode(model, readout, stamps, marks, ids, seed=314159, posterior=posterior,
                                         compiled=compiled, binding=binding, cell=cell,
                                         scale_grid=GLR_GRID if glr else None, scaled=scaled)
        pt, pg, mask = parts(ll, llt, stamps, lengths)
        total += pt.sum() + pg.sum(); count += int(mask.sum())
        rules = prefix_scores(pt, pg, lengths)
        if glr:
            sk = torch.stack([x if x is not None else torch.zeros(len(idx), len(GLR_GRID), dtype=torch.float64)
                              for x in scaled], 1).numpy()
            rules['glr_max'] = glr_prefix_scores(sk, ids.numpy(), lengths)
        out.append(rules)
        used.append(st['bseen'].sum(-1).float().mean(0, keepdim=True).numpy() if binding is not None
                    else st['seen'][-1].sum(-1).float().mean(0).numpy())
    return ({r: np.concatenate([q[r] for q in out]) for r in out[0]}, total / max(count, 1),
            dict(binding_or_top_layer_slots_written_per_run=np.mean(used, 0).round(3).tolist(),
                 pool=binding.slots if binding is not None else model.pool))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--payload', type=int, default=32); p.add_argument('--depth', type=int, default=4)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=8)
    p.add_argument('--hidden', type=int, default=64); p.add_argument('--posterior', action='store_true')
    p.add_argument('--type-durations', action='store_true', help='own durations conditioned on the next type (THEORY §435.2)')
    p.add_argument('--step-classes', type=int, default=0, help='per-slot mixture of M step classes coupling type and '
                   'duration (THEORY §436.1); cheaper than --type-durations')
    p.add_argument('--binding-slots', type=int, default=0, help='dedicated binding memory with this many slots above the deep '
                   'network, posterior-routed (THEORY §436); the readout reads it with one emitting head')
    p.add_argument('--binding-gated', action='store_true', help='per-dimension overwrite gate on binding writes (§436.2)')
    p.add_argument('--readout-context', choices=('full', 'additive', 'none'), default='full',
                   help='merged-stream context in the per-slot laws: full (inside the slot MLP), additive (shared logit '
                        'shift; per-slot laws computable once per write), none (§436.3)')
    p.add_argument('--route-credit', default='linear'); p.add_argument('--compiled', action='store_true')
    p.add_argument('--epochs', type=int, default=1); p.add_argument('--lanes', type=int, default=64)
    p.add_argument('--fit-runs', type=int, default=10000); p.add_argument('--max-events', type=int, default=1100)
    p.add_argument('--train-max-events', type=int, default=1024)
    p.add_argument('--lr', type=float, default=.003); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=6); p.add_argument('--eval-runs', type=int, default=1000)
    p.add_argument('--max-windows', type=int, default=0); p.add_argument('--segment', type=int, default=128)
    p.add_argument('--tau-max', type=float, default=1000.); p.add_argument('--no-test', action='store_true')
    p.add_argument('--cell-ms', type=float, default=0., help='recording resolution in ms for the cell likelihood (THEORY §439); '
                   '0 = point density')
    p.add_argument('--trace-windows', type=int, default=1, help='training windows traced by the operation audit (eager)')
    a = p.parse_args()
    CELL = a.cell_ms / 1000. if a.cell_ms else None
    out = Path(OUT) / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); started = time.perf_counter()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = load(d / 'train_clean.npz', a.train_max_events); train = train[:a.fit_runs]
    if len({len(r[0]) for r in train}) != 1:
        raise ValueError('equal-length training runs required: set --train-max-events <= the shortest run')
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
    binding = None
    if a.binding_slots:
        readout = RaceReadout(V, 1, a.binding_slots, a.payload, a.heads * a.payload, hidden=a.hidden, type_durations=a.type_durations,
                              classes=a.step_classes, context=dict(full=True, additive='additive', none=False)[a.readout_context])
        binding = BindingMemory(a.heads * a.payload, a.binding_slots, a.payload, tau_max=a.tau_max or 1000., gated=a.binding_gated)
    else:
        readout = RaceReadout(V, a.heads, a.pool, a.payload, a.heads * a.payload, hidden=a.hidden, type_durations=a.type_durations,
                              classes=a.step_classes)
    params = [q for q in model.parameters()] + list(readout.parameters()) + (list(binding.parameters()) if binding else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    T = len(train[0][0]); per_epoch = math.ceil(len(train) / a.lanes) * math.ceil(T / a.segment)
    total = a.epochs * per_epoch
    if a.max_windows:
        total = min(total, a.max_windows)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    rc = None if a.route_credit == 'none' else a.route_credit
    curve = []; best = (math.inf, None, None, None); events_seen = 0; w = 0; ledger = None; traced_events = 0
    from parallel_head_accumulated_language import merge
    from work_audit import capture
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(train)); t0 = time.perf_counter(); loss_sum = 0.; n_sum = 0
        for b in range(0, len(order), a.lanes):
            if w >= total:
                break
            stamps, marks, ids, _ = tensors(train, order[b:b + a.lanes])
            state = None
            for s in range(0, T, a.segment):
                if w >= total:
                    break
                sl = slice(s, s + a.segment)
                box = {}

                def train_step(compiled):
                    model.train(); readout.train(); opt.zero_grad(set_to_none=True)
                    ll, _, valid, st = readout_episode(model, readout, stamps[:, sl], marks[:, sl], ids[:, sl], state=state,
                                                       seed=100000 + w, route_credit=rc, posterior=a.posterior,
                                                       compiled=compiled, binding=binding, cell=CELL)
                    m = valid.sum()
                    loss = -(ll * valid).sum() / m.clamp_min(1)
                    loss.float().backward(); torch.nn.utils.clip_grad_norm_(params, a.clip); opt.step()
                    box.update(st=st, m=m, loss=loss)
                if w < a.trace_windows:          # measured work: eager formulation under the operation audit
                    rec = capture(lambda: train_step(False)); ledger = merge([ledger, rec]) if ledger else rec
                    traced_events += stamps.shape[0] * (min(s + a.segment, T) - s)
                else:
                    train_step(a.compiled)
                st, m, loss = box['st'], box['m'], box['loss']
                schedule.step()
                state = detach_state(st); w += 1
                events_seen += stamps.shape[0] * (min(s + a.segment, T) - s)
                loss_sum += float(loss.detach()) * int(m); n_sum += int(m)
                if w % 25 == 0:
                    print(json.dumps(dict(window=w, of=total, train_nll_t=float(loss.detach()),
                                          events_per_s=events_seen / (time.perf_counter() - started))), flush=True)
        sc_c, val_nll, occ = scores(model, readout, val_c, 64, a.posterior, a.compiled, binding, CELL)
        sc_f, _, _ = scores(model, readout, val_f, 64, a.posterior, a.compiled, binding, CELL)
        curve.append(dict(epoch=epoch, train_nll_time_units=loss_sum / max(n_sum, 1), val_clean_nll=val_nll,
                          val_auroc=aurocs(sc_c, sc_f, val_k), val_clean_occupancy=occ, epoch_s=time.perf_counter() - t0))
        print(json.dumps(curve[-1]), flush=True)
        if val_nll < best[0]:
            best = (val_nll, epoch, ({k: v.detach().clone() for k, v in model.state_dict().items()},
                                     {k: v.detach().clone() for k, v in readout.state_dict().items()},
                                     {k: v.detach().clone() for k, v in binding.state_dict().items()} if binding else None), (sc_c, sc_f))
    model.load_state_dict(best[2][0]); readout.load_state_dict(best[2][1])
    if binding is not None:
        binding.load_state_dict(best[2][2])
    weights = Path(OUT) / 'checkpoints' / f'{a.tag}_selected.pt'
    weights.parent.mkdir(parents=True, exist_ok=True); torch.save(dict(model=model.state_dict(), readout=readout.state_dict(),
                                                                   binding=binding.state_dict() if binding else None), weights)
    sc_tc = sc_tf = test_k = None; test_nll = None
    if not a.no_test:
        test_c, _ = load(d / 'test_clean.npz', a.max_events); test_f, test_k = load(d / 'test_faulty.npz', a.max_events)
        sc_tc, test_nll, _ = scores(model, readout, test_c, 64, a.posterior, a.compiled, binding, CELL)
        sc_tf, _, _ = scores(model, readout, test_f, 64, a.posterior, a.compiled, binding, CELL)
    per_run = dict(prefixes=np.array(PREFIXES), rules=np.array(RULES), val_fault_kind=np.asarray(val_k))
    for split, pair in (('val', best[3]), ('test', (sc_tc, sc_tf))):
        if pair[0] is not None:
            for r in RULES:
                per_run[f'{split}_clean_{r}'] = pair[0][r]; per_run[f'{split}_faulty_{r}'] = pair[1][r]
    score_path = out.with_name(f'{a.tag}_scores.npz'); np.savez_compressed(score_path, **per_run)
    infer = None
    if a.trace_windows:                 # measured inference work: eager, no grad, 4 validation runs
        st_i, mk_i, id_i, len_i = tensors(val_c, list(range(min(4, len(val_c)))))
        def infer_step(sparse):
            with torch.no_grad():
                readout_episode(model, readout, st_i, mk_i, id_i, seed=314159, posterior=a.posterior, binding=binding,
                                sparse=sparse, cell=CELL)
        infer = capture(lambda: infer_step(False)); infer_sparse = capture(lambda: infer_step(True))
        infer_events = st_i.shape[0] * st_i.shape[1]
    readout_params = sum(q.numel() for q in readout.parameters()) + (sum(q.numel() for q in binding.parameters()) if binding else 0)
    result = dict(status='smoke' if a.max_windows else 'completed', battle='B3', args=vars(a),
                  parameters=sum(q.numel() for q in model.parameters()) + readout_params, readout_parameters=readout_params,
                  curve=curve, selected_epoch=best[1], selection='validation-clean NLL only',
                  test_clean_nll=test_nll,
                  test_auroc=aurocs(sc_tc, sc_tf, test_k) if sc_tc is not None else 'not scored (development run, --no-test)',
                  selected_weights=_rel(weights),
                  per_run_scores=dict(path=_rel(score_path), sha256=hashlib.sha256(score_path.read_bytes()).hexdigest()),
                  work=dict(fitting_events=events_seen, traced_fitting_events=traced_events,
                            fit_arithmetic_flops_per_event=ledger['arithmetic_flops'] / traced_events if ledger else None,
                            fit_special_function_evaluations_per_event=ledger['special_function_evaluations'] / traced_events if ledger else None,
                            whole_fit_arithmetic_flops_estimate=ledger['arithmetic_flops'] / traced_events * events_seen if ledger else None,
                            inference_arithmetic_flops_per_event=infer['arithmetic_flops'] / infer_events if infer else None,
                            inference_special_function_evaluations_per_event=infer['special_function_evaluations'] / infer_events if infer else None,
                            sparse_inference_arithmetic_flops_per_event=infer_sparse['arithmetic_flops'] / infer_events if infer else None,
                            sparse_inference_special_function_evaluations_per_event=infer_sparse['special_function_evaluations'] / infer_events if infer else None,
                            scope='operation audit (experiments/fas/work_audit.py) of the first training window(s), eager, including '
                                  'backward and optimizer, extrapolated per event; inference traced on 4 validation runs '
                                  '(padded events included), dense (as trained) and sparse (winner-only deep layers with cached '
                                  'reads; identical outputs); arithmetic and special functions counted separately'),
                  data_manifest_sha256=hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest(),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/native_race_readout.py', 'experiments/fas/native.py',
                                  'sleeping_machines/race_readout.py', 'sleeping_machines/compiled_episodes.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(val=curve[-1]['val_auroc'], test_auroc=result['test_auroc'])), flush=True)


if __name__ == '__main__':
    main()
