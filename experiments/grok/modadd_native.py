"""Grokking and holography testbed (THEORY note 155, §§426–428): modular addition with the native integrated core.

Task: c = (a + b) mod p, p = 97. Each example is three events (a, b, '='), with stamps 0, 1, 2 and one-hot content
over p + 1 symbols; the prediction is read after the last event. A seeded fraction of all p^2 pairs trains; the rest
is held out.

Training: full race model with linear route credit and AdamW (default weight decay 1.0, the usual grokking setting).
Two parameter groups:
  private - parameters owned by a single unit (with --tie-pools, keys, clock biases and timescales);
  shared  - everything else (embeddings, mixing, head, and maps shared by several units under tying).
--wd-private and --wd-shared set asymmetric decay (§427.3).

Logged every --eval-every steps:
  - training and held-out accuracy (deterministic races: highest score wins);
  - the steps at which held-out accuracy first crosses .5 and .9.
At checkpoints (--holo-every):
  - holographic degree H = mean|K(x,x')| / mean K(x,x), with K the gradient inner product of the target logit
    (§426.2), on --holo-pairs training pairs;
  - the correlation over those pairs between route overlap (mean over events, layers and heads of the sum over units
    of pi_u(x) pi_u(x')) and loss-gradient alignment cos(grad L_x, grad L_x') (§428);
  - private and shared parameter norms.

Not yet available: low-rank private maps (G4 uses tied maps for the private/shared split) and k-winner races (G5).
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.carried_episodes import carried_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402

OUT = ROOT / 'experiments/results/grok'


def data(p, frac, seed):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing='ij'); a, b = a.ravel(), b.ravel()
    order = np.random.default_rng(seed).permutation(len(a)); n = int(round(frac * len(a)))
    return (a[order[:n]], b[order[:n]]), (a[order[n:]], b[order[n:]])


def events(a, b, p):
    n = len(a); V = p + 1
    marks = torch.zeros(n, 3, V)
    marks[torch.arange(n), 0, torch.from_numpy(a)] = 1; marks[torch.arange(n), 1, torch.from_numpy(b)] = 1
    marks[:, 2, p] = 1
    stamps = torch.arange(3, dtype=torch.float64).expand(n, 3).clone()
    return stamps, marks, torch.from_numpy((a + b) % p).long()


def forward(model, stamps, marks, seed, rc, deterministic, eager, collect=False):
    out = carried_logits(model, stamps, marks, seed=seed, route_credit=rc, deterministic=deterministic,
                         recruit=dict(eager=eager))
    z, _, pis = out
    return (z[:, -1], pis) if collect else z[:, -1]


def split_params(model):
    owners = {}
    for layer in model.units:
        for head in layer:
            for pool in head:
                for unit in pool:
                    for q in unit.parameters():
                        owners.setdefault(id(q), set()).add(id(unit))
    private, shared, seen = [], [], set()
    for q in model.parameters():
        if id(q) in seen:
            continue
        seen.add(id(q))
        (private if len(owners.get(id(q), ())) == 1 else shared).append(q)
    return private, shared


@torch.no_grad()
def accuracy(model, a, b, p, eager):
    model.eval(); correct = 0
    for i in range(0, len(a), 2048):
        s, m, y = events(a[i:i + 2048], b[i:i + 2048], p)
        correct += int((forward(model, s, m, 0, None, True, eager).argmax(-1) == y).sum())
    return correct / len(a)


def holography(model, a, b, p, pairs, seed, eager):
    """holographic degree, overlap/alignment correlation (§§426, 428) on `pairs` random training pairs."""
    rng = np.random.default_rng(seed); idx = rng.choice(len(a), size=min(2 * pairs, len(a)), replace=False)
    s, m, y = events(a[idx], b[idx], p)
    params = [q for q in model.parameters() if q.requires_grad]
    model.eval()
    grads_f, grads_l = [], []
    for i in range(len(idx)):
        z = forward(model, s[i:i + 1], m[i:i + 1], 0, None, True, eager)
        gf = torch.autograd.grad(z[0, y[i]], params, allow_unused=True, retain_graph=True)
        grads_f.append(torch.cat([(g if g is not None else torch.zeros_like(q)).flatten() for g, q in zip(gf, params)]))
        gl = torch.autograd.grad(F.cross_entropy(z, y[i:i + 1]), params, allow_unused=True)
        grads_l.append(torch.cat([(g if g is not None else torch.zeros_like(q)).flatten() for g, q in zip(gl, params)]))
    Gf = torch.stack(grads_f); Gl = torch.stack(grads_l)                      # float32 (memory: examples x params)
    K = (Gf @ Gf.T).double()
    off = ~torch.eye(len(idx), dtype=torch.bool)
    degree = float(K[off].abs().mean() / K.diagonal().mean())
    with torch.no_grad():
        _, pis = forward(model, s, m, 0, None, True, eager, collect=True)
    P = torch.cat([pi.reshape(len(idx), -1) for _, pi in pis], 1)          # (n, events*layers*heads*U)
    positions = sum(pi.shape[1] for _, pi in pis)
    overlap = (P @ P.T) / positions
    Gn = Gl / Gl.norm(dim=1, keepdim=True).clamp_min(1e-12); align = (Gn @ Gn.T).double()
    i, j = torch.triu_indices(len(idx), len(idx), 1)
    o, al = overlap[i, j].numpy(), align[i, j].numpy()
    corr = float(np.corrcoef(o, al)[0, 1]) if o.std() > 0 and al.std() > 0 else float('nan')
    return dict(degree=degree, overlap_alignment_corr=corr, mean_route_overlap=float(o.mean()),
                mean_alignment=float(al.mean()), pairs=int(len(i)))


def main():
    p_ = argparse.ArgumentParser(description=__doc__)
    p_.add_argument('--tag', required=True); p_.add_argument('--p', type=int, default=97)
    p_.add_argument('--train-frac', type=float, default=.4); p_.add_argument('--steps', type=int, default=20000)
    p_.add_argument('--batch', type=int, default=512); p_.add_argument('--lr', type=float, default=1e-3)
    p_.add_argument('--wd-private', type=float, default=1.); p_.add_argument('--wd-shared', type=float, default=1.)
    p_.add_argument('--payload', type=int, default=32); p_.add_argument('--depth', type=int, default=2)
    p_.add_argument('--heads', type=int, default=4); p_.add_argument('--pool', type=int, default=8)
    p_.add_argument('--tie-pools', action='store_true'); p_.add_argument('--route-credit', default='linear')
    p_.add_argument('--seed', type=int, default=0); p_.add_argument('--data-seed', type=int, default=0)
    p_.add_argument('--eval-every', type=int, default=250); p_.add_argument('--holo-every', type=int, default=2500)
    p_.add_argument('--holo-pairs', type=int, default=32)
    p_.add_argument('--post-steps', type=int, default=2000, help='stop this many steps after held-out accuracy first reaches .99 '
                    '(0: run all --steps); keeps the post-generalization compression phase'); p_.add_argument('--compiled', action='store_true')
    a = p_.parse_args()
    out = OUT / f'{a.tag}.json'
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); started = time.perf_counter()
    (ta, tb), (va, vb) = data(a.p, a.train_frac, a.data_seed)
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=a.p + 1, classes=a.p, payload=a.payload,
                                            depth=a.depth, heads=a.heads, pool=a.pool)
    if a.tie_pools:
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    private, shared = split_params(model)
    opt = torch.optim.AdamW([dict(params=private, weight_decay=a.wd_private), dict(params=shared, weight_decay=a.wd_shared)],
                            lr=a.lr, betas=(.9, .98))
    eager = not a.compiled
    rc = None if a.route_credit == 'none' else a.route_credit
    curve, holo = [], []; cross = {.5: None, .9: None, .99: None}
    for step in range(1, a.steps + 1):
        sel = rng.integers(0, len(ta), a.batch)
        s, m, y = events(ta[sel], tb[sel], a.p)
        model.train(); opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(forward(model, s, m, 100000 + step, rc, False, eager), y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        if step % a.eval_every == 0 or step == a.steps or (a.post_steps and cross.get(.99) is not None and step >= cross[.99] + a.post_steps):
            tr, te = accuracy(model, ta, tb, a.p, eager), accuracy(model, va, vb, a.p, eager)
            with torch.no_grad():
                npriv = float(torch.sqrt(sum((q ** 2).sum() for q in private))); nsh = float(torch.sqrt(sum((q ** 2).sum() for q in shared)))
            curve.append(dict(step=step, train_loss=float(loss.detach()), train_acc=tr, test_acc=te, private_norm=npriv,
                              shared_norm=nsh, wall_s=time.perf_counter() - started))
            for thr in cross:
                if cross[thr] is None and te >= thr:
                    cross[thr] = step
            print(json.dumps(curve[-1]), flush=True)
        done = a.post_steps and cross.get(.99) is not None and step >= cross[.99] + a.post_steps
        if step % a.holo_every == 0 or step == a.steps or done:
            holo.append(dict(step=step, **holography(model, ta, tb, a.p, a.holo_pairs, 7, eager)))
            print(json.dumps(holo[-1]), flush=True)
        if done:
            break
    result = dict(status='completed', args=vars(a), parameters=sum(q.numel() for q in model.parameters()),
                  private_parameters=sum(q.numel() for q in private), shared_parameters=sum(q.numel() for q in shared),
                  train_pairs=len(ta), heldout_pairs=len(va), curve=curve, holography=holo,
                  first_step_heldout_acc_ge={str(k): v for k, v in cross.items()},
                  final=dict(train_acc=curve[-1]['train_acc'], test_acc=curve[-1]['test_acc']),
                  evaluation='deterministic races (highest score wins)',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/grok/modadd_native.py', 'sleeping_machines/carried_episodes.py',
                                  'sleeping_machines/recruit_layer.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=1),
                  wall_s=time.perf_counter() - started)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(final=result['final'], cross=result['first_step_heldout_acc_ge'])), flush=True)


if __name__ == '__main__':
    main()
