"""Measure adaptive decoder branch exposure; invoke through run_safe.sh."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from horizon_token_language_engine import LaneTokens, load_tokens

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output)
if out.exists():
    raise FileExistsError(out)
torch.set_num_threads(1)
path = Path('data/fineweb_gpt2/fineweb_train_000001.bin')
data = load_tokens(path)
rows = []
for size in (2048, 8192, 32768, 65536, 262144):
    lanes = LaneTokens(data, 0, size, 8)
    counts = lanes.counts()
    order = torch.argsort(counts, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))
    targets = lanes[:, slice(1, lanes.shape[1])]
    tr = ranks[targets].flatten()
    exposure = [int(((tr >= lo) & (tr < hi)).sum())
                for lo, hi in zip((0, 2000, 10000, 30000), (2000, 10000, 30000, 50257))]
    rows.append(dict(train_tokens=size, targets=tr.numel(), distinct_tokens=int((counts > 0).sum()),
                     branch_targets=exposure, tail_targets=sum(exposure[1:])))
    assert sum(exposure) == tr.numel()
record = dict(status='completed', rows=rows, train_file=str(path),
              train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              scope='Train-only frequency rank, contiguous eight-lane next-token targets; no training or development selection. Zero tail targets means tail conditional classifiers receive no target NLL gradient; head branch gates still receive normalization gradients.')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
