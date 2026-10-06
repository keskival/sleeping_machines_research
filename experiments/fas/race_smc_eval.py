"""Particle evaluation of a race-readout FAS model (THEORY §434, Corollary 434.1.2; battle B3 development). Evaluation only.

Each validation run is evaluated with L particles (lanes) with independent race noise. Event by event:
- each particle's incremental weight is its readout likelihood p(x_k | its path);
- posterior routing (when the model was trained with --posterior) draws the particle's binding;
- when a run's effective sample size falls below L/2, its particles are resampled multinomially by their accumulated
  weights. The running estimate log Z_k = log p(x_{1:k}) adds log mean exp of the weights at each resampling.
log Z_L is the L-particle SMC estimate of the interleaving-marginal likelihood. Its expectation (the bound) tightens
as L grows (FIVO).

Score per prefix N: -log Z_N / (N - 1) (rule 'total' in log-time units; the rule ranks runs, and the units cancel in
AUROC). With L = 1 and deterministic routes, this equals the driver's own total-rule prefix NLL before the gap
Jacobian (contract: tests/test_race_readout.py::test_smc_eval_single_particle_matches_episode).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
from native import EYE, PREFIXES, V, auroc, load  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.race_readout import BindingMemory, RaceReadout, detach_state, readout_episode  # noqa: E402

OUT = ROOT / 'experiments/results/fas'


def gather_state(state, index):
    out = {}
    for k, v in state.items():
        if isinstance(v, list):
            out[k] = [x[index] for x in v]
        elif isinstance(v, dict):
            out[k] = {a: b[index] for a, b in v.items()}
        elif v is None:
            out[k] = None
        else:
            out[k] = v[index]
    return out


@torch.no_grad()
def smc_log_z(model, readout, stamps, marks, types, L, posterior, seed=314159, deterministic=False, compiled=False,
              binding=None):
    """stamps (n, T). Returns log Z (n, T): cumulative SMC log-likelihood estimate after each event (event 0 = 0)."""
    n, T = stamps.shape
    rep = lambda x: x.repeat_interleave(L, 0)
    st_s, mk_s, ty_s = rep(stamps), rep(marks), rep(types)
    state = None; logw = torch.zeros(n, L, dtype=torch.float64); logz = torch.zeros(n, dtype=torch.float64)
    out = torch.zeros(n, T, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    for k in range(T):
        ll, _, valid, st = readout_episode(model, readout, st_s[:, k:k + 1], mk_s[:, k:k + 1], ty_s[:, k:k + 1], state=state,
                                           seed=seed + 7919 * k, posterior=posterior, deterministic=deterministic,
                                           compiled=compiled, binding=binding)
        state = detach_state(st)
        logw = logw + (ll[:, 0] * valid[:, 0]).view(n, L)
        cur = logz + torch.logsumexp(logw, 1) - np.log(L)
        out[:, k] = cur
        w = torch.softmax(logw, 1); ess = 1 / (w ** 2).sum(1)
        redo = ess < L / 2
        if redo.any() and L > 1:
            idx = torch.arange(n * L).view(n, L).clone()
            pick = torch.multinomial(w[redo], L, replacement=True, generator=gen)
            idx[redo] = (torch.nonzero(redo).view(-1, 1) * L + pick)
            state = gather_state(state, idx.view(-1))
            logz = torch.where(redo, cur, logz)
            logw = torch.where(redo[:, None], torch.zeros_like(logw), logw)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True, help='completed native_race_readout.py result JSON')
    p.add_argument('--particles', default='1,4,16'); p.add_argument('--runs', type=int, default=200)
    p.add_argument('--lanes', type=int, default=16, help='runs per batch (each with L particles)')
    p.add_argument('--tag', required=True); p.add_argument('--compiled', action='store_true')
    a = p.parse_args()
    out = OUT / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1)
    res = json.loads(Path(a.result).read_text()); args = res['args']
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=V, classes=V + 2, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    binding = None
    if args.get('binding_slots'):
        readout = RaceReadout(V, 1, args['binding_slots'], args['payload'], args['heads'] * args['payload'], hidden=args['hidden'],
                              type_durations=args.get('type_durations', False), classes=args.get('step_classes', 0))
        binding = BindingMemory(args['heads'] * args['payload'], args['binding_slots'], args['payload'], tau_max=args.get('tau_max') or 1000.,
                                gated=args.get('binding_gated', False))
    else:
        readout = RaceReadout(V, args['heads'], args['pool'], args['payload'], args['heads'] * args['payload'], hidden=args['hidden'],
                              type_durations=args.get('type_durations', False), classes=args.get('step_classes', 0))
    ck = torch.load(ROOT / res['selected_weights'], weights_only=True)
    if binding is not None:
        binding.load_state_dict(ck['binding']); binding.eval()
    model.load_state_dict(ck['model']); readout.load_state_dict(ck['readout']); model.eval(); readout.eval()
    d = ROOT / 'experiments/data/fas' / args['data']
    val_c, _ = load(d / 'val_clean.npz', args['max_events']); val_f, val_k = load(d / 'val_faulty.npz', args['max_events'])
    val_c, val_f, val_k = val_c[:a.runs], val_f[:a.runs], np.asarray(val_k[:a.runs])
    from native_race_readout import tensors
    result = dict(status='completed', battle='B3', source_result=a.result, args=vars(a), posterior=args['posterior'],
                  by_particles={}, source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                                  ('experiments/fas/race_smc_eval.py', 'sleeping_machines/race_readout.py')})
    for L in map(int, a.particles.split(',')):
        t0 = time.perf_counter(); scores = {}; nll = []
        for name, runs in (('clean', val_c), ('faulty', val_f)):
            sc = np.full((len(runs), len(PREFIXES)), np.nan)
            for b in range(0, len(runs), a.lanes):
                idx = list(range(b, min(b + a.lanes, len(runs))))
                stamps, marks, ids, lengths = tensors(runs, idx)
                lz = smc_log_z(model, readout, stamps, marks, ids, L, args['posterior'] or bool(args.get('binding_slots')),
                               compiled=a.compiled, binding=binding).numpy()
                for r, n_ev in enumerate(lengths):
                    for j, N in enumerate(PREFIXES):
                        if N <= n_ev:
                            sc[b + r, j] = -lz[r, N - 1] / (N - 1)
                    if name == 'clean':
                        nll.append(-lz[r, n_ev - 1] / (n_ev - 1))
            scores[name] = sc
        row = dict(val_clean_nll_time_units=float(np.mean(nll)), wall_s=time.perf_counter() - t0, auroc={})
        for j, N in enumerate(PREFIXES):
            row['auroc'][N] = dict(all=auroc(scores['clean'][:, j], scores['faulty'][:, j]),
                                   wear_and_tear=auroc(scores['clean'][:, j], scores['faulty'][val_k == 1, j]),
                                   retry_delay=auroc(scores['clean'][:, j], scores['faulty'][val_k == 2, j]))
        result['by_particles'][L] = row
        print(json.dumps({L: row}), flush=True)
    out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
