"""R1: is the episodic store's copy gate the limit? Native-confidence gate refit on a disjoint slice (frozen arm).

The completed episodic arm loses 0.161 nats on targets that did not occur earlier in their window
(curie_episodic_bucket_gains_2m_20261015_v1.json): the gate g(x, peak score) copies with mass g(1 - null)
even when the native readout is already confident. This analysis keeps every weight of the arm frozen and
replaces only the gate by a six-parameter logistic gate on causal per-position evidence available before the
target is revealed: the learned gate's logit, native entropy, native top log-probability, the native probability
mass on the pointer's candidates (agreement), and the null weight. The gate is fitted on a DEV selection slice
disjoint from the recorded protocol and scored once on the recorded 8192 targets (1024-token windows, 8 lanes).
A gain here is the case for adding native-confidence inputs to the gate in the trained module.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit'), str(ROOT / 'analysis')]
import episodic_memory_continuation as run
from public_token_scoring import windows, reference_forward
from episodic_bucket_gains import load

SCORED, SELECT, TARGETS = 25165824, 24117248, 8192
NAMES = ('gate_logit', 'native_entropy', 'native_max_logp', 'agreement', 'null')


@torch.no_grad()
def collect(model, memory, dev, args, start):
    model.eval(); forward = reference_forward(model, run.driver); gen = torch.Generator().manual_seed(args.seed + 100000)
    d = model.core.total_payload * model.core.depth; rows = {k: [] for k in ('nat', 'ptr', 'mixed', 'seen') + NAMES}
    for inputs, labels, _ in windows(dev, start, TARGETS, 1024, 8):
        state = None; mem = memory.empty(inputs.shape[0], d); observed = [set() for _ in range(inputs.shape[0])]
        for pos in range(0, 1024, args.chunk):
            ids, y = inputs[:, pos:pos + args.chunk], labels[:, pos:pos + args.chunk]
            x, state, _ = forward(ids, state, gen, args, deterministic=args.deterministic_eval)
            nat = model.readout.nll(x, y).double().reshape(y.shape)
            mix, _, _ = memory.mixed_nll(nat.float().reshape(-1), x, ids, y, mem)
            weights, cand, null, gate, mem = memory.read(x, ids, y, mem)
            lp = model.readout.log_prob(x).reshape(*y.shape, -1).double(); p = lp.exp()
            rows['nat'].append(-nat); rows['mixed'].append(mix.double().reshape(y.shape))
            rows['ptr'].append((weights * (cand == y[..., None]).float()).sum(-1).double())
            rows['gate_logit'].append(torch.logit(gate.double().clamp(1e-6, 1 - 1e-6)))
            rows['native_entropy'].append(-(p * lp).sum(-1)); rows['native_max_logp'].append(lp.max(-1).values)
            rows['agreement'].append((weights.double() * p.gather(-1, cand)).sum(-1)); rows['null'].append(null.double())
            s = torch.zeros(y.shape, dtype=torch.bool)
            for i in range(y.shape[0]):
                for t in range(y.shape[1]): observed[i].add(int(ids[i, t])); s[i, t] = int(y[i, t]) in observed[i]
            rows['seen'].append(s)
    return {k: torch.cat([r.reshape(-1) for r in v]) for k, v in rows.items()}


def mixed_logp(theta, f, data):
    g = torch.sigmoid(theta[0] + f @ theta[1:])
    keep = torch.log((1 - g + g * data['null']).clamp_min(1e-300)) + data['nat']
    copy = torch.log((g * data['ptr']).clamp_min(1e-300))
    return torch.where(data['ptr'] > 0, torch.logaddexp(keep, copy), keep)


def features(data, mu, sd): return (torch.stack([data[k] for k in NAMES], 1) - mu) / sd


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--run', default='curie_episodic_memory_2m_episodic_20261015_v1'); a = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic()
    rec = json.loads((ROOT / f'experiments/results/token_language/{a.run}.json').read_text())
    model, args, _, memory, rec = load(a.run, rec['binding'].get('compiled', False)); dev = run.driver.load_tokens(args.dev_file)
    sel = collect(model, memory, dev, args, SELECT); sco = collect(model, memory, dev, args, SCORED)
    assert abs(sco['mixed'].mean().item() - rec['scored']['nll']) < 1e-5, (sco['mixed'].mean().item(), rec['scored']['nll'])
    mu = torch.stack([sel[k] for k in NAMES], 1).mean(0); sd = torch.stack([sel[k] for k in NAMES], 1).std(0).clamp_min(1e-8)
    fs, fc = features(sel, mu, sd), features(sco, mu, sd)
    theta = torch.zeros(len(NAMES) + 1, dtype=torch.float64); theta[1] = sd[0]   # start at the learned gate exactly
    theta[0] = mu[0]; theta.requires_grad_(True)
    assert torch.allclose(-mixed_logp(theta.detach(), fc, sco), sco['mixed'], atol=1e-4)
    opt = torch.optim.LBFGS([theta], max_iter=200, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = -mixed_logp(theta, fs, sel).mean(); loss.backward(); return loss
    opt.step(closure); theta = theta.detach()
    def summary(data, f):
        refit = -mixed_logp(theta, f, data); m = data['seen']
        out = dict(native=float(-data['nat'].mean()), learned_gate=float(data['mixed'].mean()), refit_gate=float(refit.mean()))
        for name, mask in (('seen_in_window', m), ('unseen_in_window', ~m)):
            out[name] = dict(fraction=float(m.double().mean() if name.startswith('seen') else (~m).double().mean()), native=float(-data['nat'][mask].mean()),
                             learned_gate=float(data['mixed'][mask].mean()), refit_gate=float(refit[mask].mean()))
        return out
    out = dict(status='completed', battle='R1', tag=a.tag, run=a.run, features=list(NAMES), theta=theta.tolist(), feature_mean=mu.tolist(), feature_sd=sd.tolist(),
               selection=dict(target_start=SELECT + 1, targets=TARGETS, **summary(sel, fs)), scored=dict(target_start=SCORED + 1, targets=TARGETS, **summary(sco, fc)),
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/episodic_gate_calibration.py', 'analysis/episodic_bucket_gains.py', 'experiments/episodic_memory_continuation.py', 'sleeping_machines/episodic_pointer_memory.py')},
               process_wall_s=time.monotonic() - began, scope='Frozen episodic arm; 7-parameter gate fitted on a disjoint DEV selection slice, scored once on the recorded protocol.')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(dict(scored=out['scored'], theta=[round(v, 3) for v in out['theta']])))


if __name__ == '__main__':
    main()
