#!/usr/bin/env python3
"""B4 TRAIN-only localization of the skipped longest-batch backward failure."""
import argparse
import hashlib
import json
import math
import random
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP as Stable

ROOT = adapter.ROOT
SOURCES = ['experiments/tpp/check_b4_logspread.py', 'experiments/tpp/race_tpp_logspread.py', 'experiments/tpp/race_tpp_b4g.py',
           'experiments/tpp/race_tpp_v19.py']


def summarize(x):
    finite = x[torch.isfinite(x)]
    return dict(shape=list(x.shape), nonfinite=int((~torch.isfinite(x)).sum()),
                min=float(finite.min()) if finite.numel() else None,
                max=float(finite.max()) if finite.numel() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--result', default='experiments/results/tpp_b4/b4g_github_s1.json')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    record_path = ROOT/args.result; r = json.loads(record_path.read_text())
    assert r['status'] == 'completed'
    for source, digest in r['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
    train, K = adapter.load(r['dataset'], r['split'], 'train')
    cfg = r['config']; random.seed(0); np.random.seed(0); torch.manual_seed(0)
    gaps = np.concatenate([np.diff(t) for t, m in train]); positive = gaps[gaps > 0]
    qs = np.log(np.quantile(positive, np.linspace(.1, .9, cfg['n_lognormal']))).tolist()
    edges = adapter.base.cluster_windows(positive, r['n_window'], 0) if r['n_window'] else None
    model = Stable(K+1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'],
        cfg['dv'], cfg['dropout'], r['scale'], qs, r['recording_cell'], r['n_window'], edges, cfg['state_modes'])
    # Independent finite-region contract, including logistic-window clocks.
    contract = []
    for nw in (0, 2):
        old = adapter.base.RaceTPP(3, 8, 4, 2, 2, 3, 2, 0.0, .01,
                                  [-7., -5., -3.], 1e-6, nw,
                                  [(-6., -5.), (-4., -3.)] if nw else None, 0)
        new = Stable(3, 8, 4, 2, 2, 3, 2, 0.0, .01,
                     [-7., -5., -3.], 1e-6, nw,
                     [(-6., -5.), (-4., -3.)] if nw else None, 0)
        new.load_state_dict(old.state_dict())
        hh = torch.randn(2, 7, 8); ss = torch.randn(2, 7, 3, 2)
        gaps_contract = torch.logspace(-8, 0, 14).reshape(2, 7)
        old_terms = old.clock_terms(gaps_contract, *old.clocks(hh, ss)[:4])
        new_terms = new.clock_terms(gaps_contract, *new.clocks(hh, ss)[:4])
        value_error = max(float((a-b).abs().max().detach()) for a,b in zip(old_terms,new_terms))
        sum(x.sum() for x in old_terms).backward(); sum(x.sum() for x in new_terms).backward()
        grad_error = max(float((a.grad-b.grad).abs().max())/(1+float(a.grad.abs().max()))
                         for a,b in zip(old.parameters(),new.parameters()) if a.grad is not None)
        assert value_error < 1e-9 and grad_error < 1e-10, (nw, value_error, grad_error)
        contract.append(dict(windows=nw, max_value_error=value_error, max_normalized_gradient_error=grad_error))
    checkpoint = ROOT/r['checkpoint']
    model.load_state_dict(torch.load(checkpoint, weights_only=True)); model.eval()
    batch = max(adapter.base.batches(train, cfg['batch'], False, random.Random(0)), key=lambda b:b[1].shape[1])
    t, m, mask = batch
    with torch.no_grad(): h, slots = model.encode(t, m, mask)
    hp = h[:, :-1].detach().requires_grad_(); sp = slots[:, :-1].detach().requires_grad_()
    captured = {}; error = None
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            params = model.clocks(hp, sp)
            for label, value in zip(('log_rate', 'mu', 'sigma', 'log_w', 'log_pk'), params):
                captured[label] = summarize(value.detach())
            tau = (t[:, 1:]-t[:, :-1]).clamp_min(0)
            log_h, log_s = model.clock_terms(tau, *params[:4])
            targets = m[:, 1:, None, None].expand(-1, -1, model.M, 1)
            joint = torch.logsumexp(log_h+params[4].gather(-1, targets).squeeze(-1), -1)+log_s.sum(-1)
            loss = -joint[mask[:, 1:]].mean()
            captured['head_loss_finite'] = bool(torch.isfinite(loss))
            captured['head_loss'] = float(loss.detach()) if captured['head_loss_finite'] else None
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]+[hp.grad, sp.grad]
            captured['head_gradients_finite'] = all(bool(torch.isfinite(g).all()) for g in grads)
            assert captured['head_gradients_finite']
        # Full-chain TRAIN backward also includes persistent layers and the state clock.
        model.zero_grad(set_to_none=True)
        with torch.autograd.detect_anomaly(check_nan=True):
            tl, ml, count, _ = model.loglik(t, m, mask, target_jitter=r['recording_cell'])
            full_loss = -(tl+ml)/count
            assert bool(torch.isfinite(full_loss))
            full_loss.backward()
            captured['full_loss'] = float(full_loss.detach())
            captured['full_gradients_finite'] = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            assert captured['full_gradients_finite']
    except RuntimeError as exception:
        error = str(exception)
        print(traceback.format_exc(), flush=True)
    captured['failed_backward'] = error
    # A bounded diagnosis is completed even when it reproduces the expected failure.
    result = dict(status='completed' if error is None else 'failed', contract=contract, tag=args.tag, battle='B4 development', stage='logspread_contract_real_backward',
                  dataset=r['dataset'], split=r['split'], shape=list(m.shape), train_events=int(mask.sum()),
                  zero_train_gaps=int(((t[:, 1:]-t[:, :-1])==0)[mask[:, 1:]].sum()),
                  captured=captured, head_backward_failure_reproduced=error is not None,
                  checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  source_result=dict(path=args.result, sha256=hashlib.sha256(record_path.read_bytes()).hexdigest()),
                  data_sha256=hashlib.sha256((adapter.DATA/r['dataset']/f"split_{r['split']}"/'train.json').read_bytes()).hexdigest(),
                  wall_s=time.monotonic()-start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
                  scope='Finite-region value/gradient contracts plus longest TRAIN batch at saved checkpoint; dropout off, head and full-chain backwards. '
                        'No optimizer update, no TEST loading, no changed benchmark score.')
    out = ROOT/'experiments/results/tpp_b4'/f'{args.tag}.json'
    with out.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
