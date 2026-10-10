#!/usr/bin/env python3
"""tgbl-review causal driver v4 = v3 (= v2 + amortized sampler) with two model switches for the measured failure.

Failure (v3 full development epoch, 10 Oct): val MRR 0.268 < training-free 30-day popularity 0.340; known-source queries
0.247 < new-source queries 0.475, so source-specific parameters damage ranking. Switches (defaults reproduce v3 exactly):
  --identity full|bias|none   full = es(s)·ed(c) + bd(c) (v3); bias = bd(c) only; none = no identity parameters
  --pop-residual              score = a·log1p(pop_30d) + network, a initialized to 1: the model starts AT the
                              training-free heuristic and learns a residual (note: the network still sees all features)
Causal state, features, competitors, losing-candidate credit and the official Evaluator are unchanged. No TEST flag.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import race_link_review_v3 as v3  # noqa: E402  (installs the amortized sampler)
import race_link_review as base  # noqa: E402

OPT = dict(identity='full', pop_residual=False)
POP30 = 9                                   # feature index: log1p(destination popularity decayed at 30 days)
_BaseModel = base.Model


class Model(_BaseModel):
    def __init__(self, n, dim, hidden):
        super().__init__(n, dim, hidden)
        if OPT['identity'] == 'none':
            for e in (self.es, self.ed, self.bd):
                e.weight.requires_grad_(False); nn.init.zeros_(e.weight)
        elif OPT['identity'] == 'bias':
            for e in (self.es, self.ed):
                e.weight.requires_grad_(False); nn.init.zeros_(e.weight)
        self.pop_scale = nn.Parameter(torch.tensor(1.0)) if OPT['pop_residual'] else None

    def forward(self, s, cand, feat):
        x = self.mlp(self.norm(feat)).squeeze(-1)
        if OPT['identity'] == 'full':
            x = x + (self.es(s)[:, None] * self.ed(cand)).sum(-1) + self.bd(cand).squeeze(-1)
        elif OPT['identity'] == 'bias':
            x = x + self.bd(cand).squeeze(-1)
        if self.pop_scale is not None:
            x = x + self.pop_scale * feat[..., POP30]
        return x


base.Model = Model


def main():
    if '--identity' in sys.argv:
        i = sys.argv.index('--identity'); OPT['identity'] = sys.argv[i + 1]; del sys.argv[i:i + 2]
        assert OPT['identity'] in ('full', 'bias', 'none')
    if '--pop-residual' in sys.argv:
        sys.argv.remove('--pop-residual'); OPT['pop_residual'] = True
    tag = sys.argv[sys.argv.index('--tag') + 1]
    v3.main()                                  # v3 writes the result and its own equivalence fields
    out = v3.v2.ROOT / 'experiments/results/tgb' / f'{tag}.json'
    res = json.loads(out.read_text())
    res['driver'] = 'race_link_review_v4.py (v3 + identity/pop-residual switches)'
    res['model_switches'] = dict(OPT)
    res.setdefault('source_sha256', {})[str(Path(__file__).resolve().relative_to(v3.v2.ROOT))] = \
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
