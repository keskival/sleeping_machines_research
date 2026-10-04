"""NeuroBench Mackey-Glass with the exact-expected-reception member (sleeping_machines/expected_reception.py, THEORY §418).

Protocol, data slices, targets and SMAPE come unchanged from experiments/mackey_glass_native.py (official neurobench
2.3.0 protocol; contract-tested).  The model is the integrated core with deterministic exact expected delivery at every
receiver head, the race's mean clock and hard argmax writes; trained teacher-forced on increment targets (or values);
forecast autonomously with ExpectedStepper.  Development uses tau 18/19; tau 17 is scored once with fixed settings.
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
import mackey_glass_native as M  # noqa: E402
from sleeping_machines.expected_reception import ExpectedStepper, expected_logits  # noqa: E402


def fit_and_forecast(a, z, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 1)
    model = M.make_model(a)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    S, B = a.segment, a.lanes
    losses = []
    for step in range(a.steps):
        starts = rng.integers(0, M.TRAIN - S + 1, B)
        rows = [dict(events=[(float(i), M.taps_of(z, i, a.taps)) for i in range(s, s + S)]) for s in starts]
        y = torch.tensor(np.stack([z[s + 1:s + S + 1] for s in starts]), dtype=torch.float32)
        if a.delta:
            y = y - torch.tensor(np.stack([z[s:s + S] for s in starts]), dtype=torch.float32)
        model.train(); opt.zero_grad(set_to_none=True)
        out = expected_logits(model, rows, 0, all_logits=True, compiled=a.compiled)[..., 0]
        loss = F.mse_loss(out[:, a.warmup:], y[:, a.warmup:])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); schedule.step()
        losses.append(float(loss.detach()))
    model.eval()
    stepper = ExpectedStepper(model, 1)
    with torch.no_grad():
        for i in range(M.TRAIN):
            pred = float(stepper.step(torch.tensor([float(i)]), torch.tensor(M.taps_of(z, i, a.taps))[None])[0, 0])
        if a.delta:
            pred += float(z[M.TRAIN - 1])
        history = list(z[:M.TRAIN]); preds = []
        for i in range(M.TRAIN, M.TRAIN + M.TEST):
            history.append(pred)
            pred = float(stepper.step(torch.tensor([float(i)]), torch.tensor(M.taps_of(np.array(history), i, a.taps))[None])[0, 0])
            if a.delta:
                pred += history[-1]
            preds.append(pred)
    return np.array(preds), losses, sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--tau', type=int, default=17)
    p.add_argument('--repeats', type=int, default=30); p.add_argument('--first-repeat', type=int, default=0)
    p.add_argument('--payload', type=int, default=16); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--tie-pools', action='store_true'); p.add_argument('--taps', type=int, default=8)
    p.add_argument('--segment', type=int, default=64); p.add_argument('--lanes', type=int, default=32)
    p.add_argument('--steps', type=int, default=1500); p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--weight-decay', type=float, default=0.); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--warmup', type=int, default=8); p.add_argument('--delta', action='store_true')
    p.add_argument('--compiled', action='store_true'); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = ROOT / 'experiments/results/neurobench_mg' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); started = time.perf_counter(); rows = []
    for r in range(a.first_repeat, a.first_repeat + a.repeats):
        raw = M.official_slice(a.tau, r)
        mu, sd = raw[:M.TRAIN + 1].mean(), raw[:M.TRAIN + 1].std()
        z = ((raw - mu) / sd).astype(np.float32)
        t = time.perf_counter()
        preds, losses, params = fit_and_forecast(a, z, a.seed * 1000 + r)
        score = M.smape(preds * sd + mu, raw[M.TRAIN + 1:M.TRAIN + M.TEST + 1])
        rows.append(dict(repeat=r, smape=score, final_train_mse=float(np.mean(losses[-20:])), wall_s=time.perf_counter() - t))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(status='completed', args=vars(a), parameters=params, footprint_bytes_float32=params * 4, repeats=rows,
                  mean_smape=float(np.mean([r['smape'] for r in rows])),
                  member='exact expected reception, mean clock, hard argmax writes (deterministic)',
                  protocol='NeuroBench Mackey-Glass (neurobench 2.3.0 slices, split and SMAPE via mackey_glass_native)',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/mg_expected.py', 'experiments/mackey_glass_native.py',
                                  'sleeping_machines/expected_reception.py', 'sleeping_machines/compiled_episodes.py')},
                  data_sha256=hashlib.sha256((M.DATA / f'mg_{a.tau}.npy').read_bytes()).hexdigest(),
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mean_smape=result['mean_smape'], parameters=params)), flush=True)


if __name__ == '__main__':
    main()
