"""Spiking Heidelberg Digits with the integrated native core (experiments/SOTA_TARGETS.md, target 3).

Official data (zenkelab.org/datasets): shd_train.h5 (8,156 utterances) and shd_test.h5 (2,264; two speakers occur only in
the test set), 700 input channels, 20 classes.  Development: speakers --val-speakers of the training file are held out for
epoch selection (the official test file is only scored with the selected weights).  Events: one per time bin of --bin-ms
(timestamp = bin index; content = log1p spike counts of the bin, channels summed in groups of --channel-group).
Classification: cross-entropy on the logits averaged over all events of an utterance.  Model: the integrated core with
route credit, compiled training, exact winner-only evaluation is equivalent to the batched evaluation used here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402

DATA = ROOT / 'data/shd'


def load(split, bin_ms, group, max_time=1.4):
    with h5py.File(DATA / f'shd_{split}.h5', 'r') as f:
        times, units = f['spikes']['times'], f['spikes']['units']
        labels = np.array(f['labels']); speakers = np.array(f['extra']['speaker'])
        nb = int(np.ceil(max_time * 1000 / bin_ms)); C = 700 // group
        X = []
        for i in range(len(labels)):
            t, u = np.asarray(times[i]), np.asarray(units[i])
            b = np.minimum((t * 1000 / bin_ms).astype(np.int64), nb - 1)
            counts = np.zeros((nb, C), np.float32)
            np.add.at(counts, (b, u // group), 1.)
            last = int(b.max()) + 1 if len(b) else 1
            X.append(np.log1p(counts[:last]))
    return X, labels.astype(np.int64), speakers


def rows_of(X, idx, noise=0., rng=None):
    rows = []
    for i in idx:
        x = X[i]
        if noise and rng is not None:
            x = x + rng.normal(0, noise, x.shape).astype(np.float32)
        rows.append(dict(events=[(float(t), x[t]) for t in range(len(x))]))
    return rows


def mean_logits(z, rows):
    lengths = torch.tensor([len(r['events']) for r in rows], dtype=z.dtype)
    return z.sum(1) / lengths[:, None]


def evaluate(model, X, y, idx, lanes):
    model.eval(); correct = 0; nll = 0.
    with torch.no_grad():
        for b in range(0, len(idx), lanes):
            chunk = idx[b:b + lanes]; rows = rows_of(X, chunk)
            logits = mean_logits(batched_logits(model, rows, 314159, all_logits=True), rows)
            target = torch.tensor(y[chunk])
            correct += int((logits.argmax(-1) == target).sum()); nll += float(F.cross_entropy(logits, target, reduction='sum'))
    return correct / len(idx), nll / len(idx)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--bin-ms', type=float, default=10.)
    p.add_argument('--channel-group', type=int, default=5); p.add_argument('--val-speakers', default='3,6')
    p.add_argument('--payload', type=int, default=32); p.add_argument('--depth', type=int, default=4)
    p.add_argument('--heads', type=int, default=2); p.add_argument('--pool', type=int, default=2)
    p.add_argument('--tie-pools', action='store_true'); p.add_argument('--route-credit', default='linear')
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--lanes', type=int, default=32)
    p.add_argument('--lr', type=float, default=.003); p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--input-noise', type=float, default=0.); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=6); p.add_argument('--compiled', action='store_true')
    p.add_argument('--official-test', action='store_true', help='score the official test file with the selected weights')
    a = p.parse_args()
    out = ROOT / 'experiments/results/shd_native' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); started = time.perf_counter()
    X, y, spk = load('train', a.bin_ms, a.channel_group)
    held = np.isin(spk, [int(s) for s in a.val_speakers.split(',')])
    fit_idx, val_idx = np.where(~held)[0], np.where(held)[0]
    C = X[0].shape[1]
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=C, classes=20, payload=a.payload, depth=a.depth,
                                            heads=a.heads, pool=a.pool)
    if a.tie_pools:
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * int(np.ceil(len(fit_idx) / a.lanes))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rc = None if a.route_credit == 'none' else a.route_credit
    logits_fn = batched_logits
    if a.compiled:
        from sleeping_machines.compiled_episodes import compiled_logits
        logits_fn = compiled_logits
    curve = []; best = (-1, None, None); step = 0
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(fit_idx); t0 = time.perf_counter(); loss_sum = 0.
        for b in range(0, len(order), a.lanes):
            chunk = order[b:b + a.lanes]; rows = rows_of(X, chunk, a.input_noise, rng)
            model.train(); opt.zero_grad(set_to_none=True)
            logits = mean_logits(logits_fn(model, rows, 100000 + step, all_logits=True, route_credit=rc), rows)
            loss = F.cross_entropy(logits, torch.tensor(y[chunk]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); schedule.step()
            loss_sum += float(loss.detach()) * len(chunk); step += 1
        acc, nll = evaluate(model, X, y, val_idx, 64)
        curve.append(dict(epoch=epoch, train_loss=loss_sum / len(order), val_accuracy=acc, val_nll=nll,
                          epoch_s=time.perf_counter() - t0))
        print(json.dumps(curve[-1]), flush=True)
        if acc > best[0]:
            best = (acc, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    result = dict(status='completed', args=vars(a), parameters=sum(q.numel() for q in model.parameters()), curve=curve,
                  selected_epoch=best[1], val_accuracy=best[0], fit_utterances=len(fit_idx), val_utterances=len(val_idx),
                  protocol='SHD official files; speaker-held-out validation from the training file for selection',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/shd_native.py', 'sleeping_machines/batched_episodes.py',
                                  'sleeping_machines/compiled_episodes.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1))
    if a.official_test:
        model.load_state_dict(best[2])
        Xt, yt, _ = load('test', a.bin_ms, a.channel_group)
        result['test_accuracy'], result['test_nll'] = evaluate(model, Xt, yt, np.arange(len(yt)), 64)
    result['wall_s'] = time.perf_counter() - started
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('val_accuracy', 'selected_epoch', 'test_accuracy')}), flush=True)


if __name__ == '__main__':
    main()
