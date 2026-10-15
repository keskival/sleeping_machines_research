"""R1 long-range addressed copy-memory opportunity on the strongest native learner.

Frozen weights, same 8192 DEV targets / disjoint 1024-token windows / no EOS reset
as the recorded score. Adds causal in-window caches (decayed unigram recency and
induction: tokens that followed earlier occurrences of the current one/two-token
context) by linear probability mixing. Mixture weights and decays are cross-fitted
on alternating windows, so every reported gain is held out. No fitting of the
network, no public validation.
"""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import protected_depth_local_gate as gate
from public_token_scoring import windows, reference_forward

LEARNER = 'experiments/results/token_language/aws_memory_policy_corpus_eos_2m_20261015_v1'
START, TARGETS, LENGTH, LANES = 25165824, 8192, 1024, 8
DECAYS = (32., 128., 512., 1e9)
RANKS = ((0, 2000), (2000, 10000), (10000, 30000), (30000, 50257))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def native_losses(model, dev, args):
    model.eval(); forward = reference_forward(model, gate.pilot.driver)
    gen = torch.Generator().manual_seed(args.seed + 100000); rows = []
    for inputs, labels, _ in windows(dev, START, TARGETS, LENGTH, LANES):
        state = None; out = []
        for pos in range(0, LENGTH, args.chunk):
            x, state, _ = forward(inputs[:, pos:pos + args.chunk], state, gen, args, deterministic=args.deterministic_eval)
            out.append(model.readout.nll(x, labels[:, pos:pos + args.chunk]).double().reshape(inputs.shape[0], -1))
        rows += [(inputs[i].numpy(), labels[i].numpy(), torch.cat(out, 1)[i].numpy()) for i in range(inputs.shape[0])]
    return rows


def caches(inputs, labels, decay):
    """Causal cache probabilities of each target; nan where the cache is empty."""
    n = len(labels); uni = np.full(n, np.nan); ind1 = np.full(n, np.nan); ind2 = np.full(n, np.nan)
    seq = np.concatenate([inputs, labels[-1:]])  # seq[t+1] == labels[t]
    for t in range(n):
        y = labels[t]; hist = seq[:t + 1]  # tokens observed before predicting labels[t]
        ages = t - np.arange(t + 1)
        weights = np.exp(-ages / decay)
        tot = weights.sum(); uni[t] = weights[hist == y].sum() / tot
        if t >= 1:
            prev = np.nonzero(seq[:t] == seq[t])[0]  # earlier occurrences of current token, followed by seq[i+1]
            if len(prev):
                wt = np.exp(-(t - prev) / decay); ind1[t] = wt[seq[prev + 1] == y].sum() / wt.sum()
                match2 = prev[(prev >= 1)]
                match2 = match2[seq[match2 - 1] == seq[t - 1]]
                if len(match2):
                    wt = np.exp(-(t - match2) / decay); ind2[t] = wt[seq[match2 + 1] == y].sum() / wt.sum()
    return uni, ind1, ind2


def mix_loss(pn, comps, lams):
    p = pn * (1 - sum(lams))
    for c, lam in zip(comps, lams):
        p = p + lam * np.where(np.isnan(c), pn, c)  # empty cache returns its mass to the network
    return -np.log(p)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--tag', required=True); args_cli = a.parse_args()
    torch.set_num_threads(1); began = time.monotonic()
    driver = gate.pilot.driver
    record = json.loads((ROOT / 'experiments/results/token_language/aws_multidepth_readout_8m_20261013_s6_v1.json').read_text())
    from types import SimpleNamespace
    args = SimpleNamespace(**record['args']); train = driver.load_tokens(args.train_file)
    counts = driver.LaneTokens(train, 0, args.train_tokens, args.lanes).counts(); order = torch.argsort(counts, descending=True, stable=True)
    torch.manual_seed(args.seed); model = driver.Model(args, order, counts)
    ck = torch.load(ROOT / (LEARNER + '.pt'), weights_only=False); model.load_state_dict(ck['model'])
    for name, rng in ck['owned_rng'].items():
        if hasattr(model, name): getattr(model, name).set_state(rng)
    dev = gate.pilot.driver.load_tokens(args.dev_file)
    rows = native_losses(model, dev, args)
    inputs = np.stack([r[0] for r in rows]); labels = np.stack([r[1] for r in rows]); ln = np.stack([r[2] for r in rows])
    recorded = json.loads((ROOT / (LEARNER + '.json')).read_text())['scored']['nll']
    assert abs(ln.mean() - recorded) < 1e-6, (ln.mean(), recorded)
    pn = np.exp(-ln)
    counts = gate.pilot.driver.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
    rank = torch.empty_like(counts); rank[torch.argsort(counts, descending=True, stable=True)] = torch.arange(len(counts))
    rank = rank.numpy()[labels]
    seen = np.zeros_like(labels, dtype=bool)
    for i in range(len(labels)):
        seq = np.concatenate([inputs[i], labels[i][-1:]])
        observed = set()
        for t in range(LENGTH):
            observed.add(seq[t]); seen[i, t] = labels[i][t] in observed
    comp = {d: [np.stack(z) for z in zip(*[caches(inputs[i], labels[i], d) for i in range(len(labels))])] for d in DECAYS}
    grid = [x / 100 for x in (0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30)]
    folds = [np.arange(len(labels)) % 2 == f for f in (0, 1)]
    held = np.zeros_like(ln); choices = []
    for f in (0, 1):
        fit, test = folds[1 - f], folds[f]; best = (math.inf, None)
        for d in DECAYS:
            u, i1, i2 = comp[d]
            for lu in grid:
                for l1 in grid:
                    for l2 in grid:
                        if lu + l1 + l2 > 0.6: continue
                        v = mix_loss(pn[fit], [u[fit], i1[fit], i2[fit]], (lu, l1, l2)).mean()
                        if v < best[0]: best = (v, (d, lu, l1, l2))
        d, lu, l1, l2 = best[1]; u, i1, i2 = comp[d]
        held[test] = mix_loss(pn[test], [u[test], i1[test], i2[test]], (lu, l1, l2)); choices.append(dict(fold=f, decay=d, unigram=lu, induction1=l1, induction2=l2, fit_nll=best[0]))
    buckets = []
    for lo, hi in RANKS:
        m = (rank >= lo) & (rank < hi)
        buckets.append(dict(rank_low=lo, rank_high=hi, target_fraction=float(m.mean()), native_nll=float(ln[m].mean()), with_cache_nll=float(held[m].mean()),
                            gain=float(ln[m].mean() - held[m].mean()), seen_in_window_fraction=float(seen[m].mean())))
    record = dict(status='completed', battle='R1', tag=args_cli.tag, learner=LEARNER, native_nll=float(ln.mean()), recorded_nll=recorded,
                  heldout_with_cache_nll=float(held.mean()), heldout_gain=float(ln.mean() - held.mean()),
                  seen_in_window_fraction=float(seen.mean()), native_nll_seen=float(ln[seen].mean()), native_nll_unseen=float(ln[~seen].mean()),
                  cache_nll_seen=float(held[seen].mean()), cache_nll_unseen=float(held[~seen].mean()),
                  buckets=buckets, crossfit_choices=choices,
                  protocol=dict(targets=TARGETS, target_start=START + 1, window=LENGTH, lanes=LANES, eos_state_reset=False, folds='alternating 1024-token windows'),
                  source_sha256={p: sha(ROOT / p) for p in ('analysis/copy_memory_opportunity.py', 'experiments/public_token_scoring.py', 'experiments/credit/protected_depth_local_gate.py')},
                  input_sha256={LEARNER + '.pt': sha(ROOT / (LEARNER + '.pt'))}, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  process_wall_s=time.monotonic() - began,
                  scope='Frozen strongest native learner; fixed non-learned causal in-window caches mixed linearly, weights/decay cross-fitted on alternating windows. Measures opportunity for a learned long-range addressed copy memory; not a trained model, public score or quality promotion.')
    out = ROOT / f'experiments/results/diagnostics/{args_cli.tag}.json'
    with out.open('x') as fh: json.dump(record, fh, indent=2); fh.write('\n')
    print(json.dumps({k: record[k] for k in ('native_nll', 'heldout_with_cache_nll', 'heldout_gain', 'seen_in_window_fraction')}))


if __name__ == '__main__':
    main()
