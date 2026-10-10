"""B4 own-model numerical development: gated TRAIN/DEV fit, no TEST interface."""
import argparse
import hashlib
import json
import math
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP

ROOT = adapter.ROOT
SOURCES = ['experiments/tpp/b4_logspread_develop.py', 'experiments/tpp/race_tpp_logspread.py',
           'experiments/tpp/race_tpp_b4g.py', 'experiments/tpp/race_tpp_v19.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    p.add_argument('--gate', required=True)
    a = p.parse_args()
    gate_path = ROOT/a.gate; gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'completed' and gate['recovered_update_gate']
    for source, sha in gate['source_sha256'].items():
        assert digest(ROOT/source) == sha, source
    pilot_path = ROOT/gate['input']['path']; pilot = json.loads(pilot_path.read_text())
    assert digest(pilot_path) == gate['input']['sha256']
    for source, sha in pilot['source_sha256'].items():
        assert digest(ROOT/source) == sha, source
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); rng = random.Random(0)
    cfg = pilot['config']; start = time.monotonic()
    train, K = adapter.load('github_filtered', 1, 'train')
    val, _ = adapter.load('github_filtered', 1, 'val')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(.1, .9, cfg['n_lognormal']))).tolist()
    model = RaceTPP(K+1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'],
                    cfg['dv'], cfg['dropout'], pilot['scale'], qs, pilot['recording_cell'],
                    pilot['n_window'], None, cfg['state_modes'])
    assert pilot['n_window'] == 0 # This admission is Github split1, not a new grid.
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.0)
    out = ROOT/'experiments/results/tpp_b4'; result_path = out/f'{a.tag}.json'
    assert not result_path.exists()
    best_path = out/f'{a.tag}.pt'; resume_path = out/f'{a.tag}.resume.pt'
    assert not best_path.exists() and not resume_path.exists()
    history = []; skips = []; best = math.inf; best_epoch = -1
    train_targets = sum(len(m)-1 for _, m in train); train_s = eval_s = 0.0
    stop = '100-pass budget'; failure = None
    for epoch in range(100):
        tick = time.monotonic()
        adapter.train_epoch(model, opt, train, cfg, rng, pilot['recording_cell'], skips, epoch)
        elapsed = time.monotonic()-tick; train_s += elapsed
        if skips:
            failure = 'Nonfinite update returned after passed pilot'; stop = 'finite-update gate failed'; break
        tick = time.monotonic(); v = adapter.per_sequence(model, val); evaluation = time.monotonic()-tick; eval_s += evaluation
        if not all(math.isfinite(x) for x in v.values()):
            failure = 'Nonfinite DEV score'; stop = 'finite-quality gate failed'; break
        if v['total'] < best:
            best = v['total']; best_epoch = epoch; torch.save(model.state_dict(), best_path)
        row = dict(epoch=epoch, val=v, train_s=elapsed, evaluation_s=evaluation,
                   scored_train_targets=train_targets, skipped_updates=len(skips))
        history.append(row); print(json.dumps(row), flush=True)
        tmp = resume_path.with_suffix('.tmp')
        torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                        numpy_rng=np.random.get_state(), batch_rng=rng.getstate(), history=history), tmp)
        tmp.replace(resume_path)
        if epoch-best_epoch >= 30: stop = 'original 30-pass DEV patience'; break
        if time.monotonic()-start > 7200: stop = '7200-second between-pass wall budget'; break
    # Inference measurement uses the selected checkpoint and DEV, never TEST.
    inference = None
    if failure is None:
        model.load_state_dict(torch.load(best_path, weights_only=True)); model.eval()
        tick = time.monotonic(); count = 0
        with torch.no_grad():
            for batch in adapter.base.batches(val, cfg['batch'], False, random.Random(0)):
                _, _, n, _ = model.loglik(*batch); count += int(n)
        seconds = time.monotonic()-tick
        inference = dict(protocol='full DEV likelihood forward, one CPU thread float64',
                         scored_targets=count, wall_s=seconds, seconds_per_target=seconds/count)
    result = dict(status='completed' if failure is None else 'failed', tag=a.tag, battle='B4 development',
                  stage='logspread_development_fit', dataset='github_filtered', split=1, seed=0, config=cfg,
                  epochs_run=len(history), best_epoch=best_epoch, best_dev_total=best if math.isfinite(best) else None,
                  history=history, skipped_updates=len(skips), skipped_first=skips[:20], failure=failure,
                  stop=stop, train_targets_per_pass=train_targets, completed_train_targets=train_targets*len(history),
                  fitting_wall_s=train_s, evaluation_wall_s=eval_s, wall_s=time.monotonic()-start,
                  inference=inference, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hardware=dict(host=platform.node(), threads=1, dtype='float64', processor=platform.processor()),
                  best_checkpoint=str(best_path.relative_to(ROOT)) if best_path.exists() else None,
                  resume_checkpoint=str(resume_path.relative_to(ROOT)) if resume_path.exists() else None,
                  gate=dict(path=a.gate, sha256=digest(gate_path)), source_sha256={s:digest(ROOT/s) for s in SOURCES},
                  data_sha256={part:digest(adapter.DATA/'github_filtered'/'split_1'/f'{part}.json') for part in ('train','val')},
                  scope='One fixed seed/split numerical development fit, original settings and patience. No TEST or frozen benchmark replacement. Wall time measured; full fitting FLOPs unmeasured.')
    result_path.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
