"""R1 residual race stack contracts and initial signal propagation (no fitting).

1. Residual off: the residual driver reproduces the original all-depth driver bit for bit
   (features, numeric state, loss) on real tokens.
2. Signal propagation at initialization, depth 8 and 16, serial vs residual: per-depth
   race-bank gradient norms from a fixed random cotangent on the deepest output only, and
   the readout-input energy share contributed by each depth.
"""
import argparse, json, sys, time, hashlib, copy
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch
import multidepth_readout_token_language as serial_driver
import residual_depth_token_language as residual_driver


def config(depth, residual):
    a = SimpleNamespace(**json.loads((ROOT / 'experiments/results/token_language/aws_multidepth_readout_8m_20261013_s6_v1.json').read_text())['args'])
    a.depth = depth; a.residual = residual; a.residual_init = None; a.compiled = False
    return a


def build(driver, a, counts, order):
    torch.manual_seed(a.seed); return driver.Model(a, order, counts)


def bank_depth_norms(model, depth):
    out = {}
    for name, p in model.core.named_parameters():
        if name.startswith('banks.') and p.grad is not None and p.grad.shape[0] == depth:
            out[name.split('.')[1]] = [float(p.grad[d].norm()) for d in range(depth)]
    total = [sum(v[d] ** 2 for v in out.values()) ** .5 for d in range(depth)]
    return out, total


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a0 = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    base = config(2, False); train = serial_driver.load_tokens(base.train_file)
    counts = serial_driver.LaneTokens(train, 0, base.train_tokens, base.lanes).counts(); order = torch.argsort(counts, descending=True, stable=True)
    ids = torch.from_numpy(train[1_000_000:1_000_000 + 8 * 65].astype('int64')).reshape(8, 65)
    out = {}
    # 1. Residual off equals the original driver exactly.
    for depth in (2, 8):
        a = config(depth, False); m1 = build(serial_driver, a, counts, order); m2 = build(residual_driver, a, counts, order)
        m2.load_state_dict(m1.state_dict())
        g1 = torch.Generator().manual_seed(1); g2 = torch.Generator().manual_seed(1)
        x1, s1, _ = m1(ids[:, :-1], None, g1, a); x2, s2, _ = m2(ids[:, :-1], None, g2, a)
        l1 = m1.readout.nll(x1, ids[:, 1:]).mean(); l2 = m2.readout.nll(x2, ids[:, 1:]).mean()
        out[f'residual_off_equal_d{depth}'] = bool(torch.equal(x1, x2) and torch.equal(l1, l2) and all(torch.equal(s1[k][i], s2[k][i]) if isinstance(s1[k], list) else torch.equal(s1[k], s2[k]) for k in ('mem', 'arr', 'seen') for i in range(depth)))
    # 2. Initial signal propagation.
    prop = {}
    for depth in (8, 16):
        for residual in (False, True):
            a = config(depth, residual); m = build(residual_driver, a, counts, order); m.train()
            g = torch.Generator().manual_seed(1)
            x, _, _ = m(ids[:, :-1], None, g, a); loss = m.readout.nll(x, ids[:, 1:]).mean()
            # Frequency-initialized readouts start with zero feature weights, so probe the stack with a fixed
            # random cotangent on the DEEPEST layer's output slice only (reaches layer 0 through every later stage).
            width = a.payload * a.heads; cot = torch.zeros_like(x)
            cot[..., (depth - 1) * width:] = torch.randn(x.shape[:-1] + (width,), generator=torch.Generator().manual_seed(7))
            x.backward(cot)
            per_bank, total = bank_depth_norms(m, depth)
            feats = x.detach().reshape(-1, depth, width)
            share = (feats.square().sum((0, 2)) / feats.square().sum()).tolist()
            prop[f'd{depth}_{"residual" if residual else "serial"}'] = dict(initial_loss=float(loss), bank_gradient_norm_by_depth=total,
                first_to_last_gradient_ratio=total[0] / max(total[-1], 1e-30), readout_feature_energy_share_by_depth=share)
    out['propagation'] = prop
    # 3. Content retention at initialization: held-out linear decodability of the current token from each depth's output.
    long_ids = torch.from_numpy(train[2_000_000:2_000_000 + 8 * 257].astype('int64')).reshape(8, 257)
    retention = {}
    for depth in (8, 16):
        for residual in (False, True):
            a = config(depth, residual); m = build(residual_driver, a, counts, order); m.eval()
            with torch.no_grad():
                x, _, _ = m(long_ids[:, :-1], None, torch.Generator().manual_seed(1), a)
            width = a.payload * a.heads; feats = x.reshape(-1, depth, width).double(); tok = long_ids[:, :-1].reshape(-1)
            common = torch.unique(tok, return_counts=True); keep_tokens = common[0][common[1] >= 4]
            mask = torch.isin(tok, keep_tokens); cls = torch.searchsorted(keep_tokens, tok[mask])
            half = torch.arange(int(mask.sum())) % 2 == 0; acc = []
            for d in range(depth):
                f = feats[mask, d]; f = (f - f[half].mean(0)) / f[half].std(0).clamp_min(1e-8)
                f = torch.cat([f, torch.ones(len(f), 1, dtype=f.dtype)], 1)
                y = torch.nn.functional.one_hot(cls, len(keep_tokens)).double()
                w = torch.linalg.solve(f[half].T @ f[half] + 1e-2 * torch.eye(f.shape[1], dtype=f.dtype), f[half].T @ y[half])
                acc.append(float(((f[~half] @ w).argmax(1) == cls[~half]).double().mean()))
            retention[f'd{depth}_{"residual" if residual else "serial"}'] = dict(heldout_token_accuracy_by_depth=acc, classes=len(keep_tokens), positions=int(mask.sum()))
    out['content_retention'] = retention
    out['source_sha256'] = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('experiments/residual_depth_contracts.py', 'experiments/residual_depth_token_language.py', 'sleeping_machines/residual_race_episodes.py', 'sleeping_machines/residual_ordered_episodes.py')}
    out['status'] = 'completed' if out['residual_off_equal_d2'] and out['residual_off_equal_d8'] else 'failed'
    out['battle'] = 'R1'; out['process_wall_s'] = time.monotonic() - began
    with (ROOT / f'experiments/results/diagnostics/{a0.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in out.items() if k.startswith('residual_off')}))
    for k, v in out['content_retention'].items(): print(k, 'token decodability by depth', [round(x, 3) for x in v['heldout_token_accuracy_by_depth']], v['classes'])
    for k, v in prop.items(): print(k, 'loss', round(v['initial_loss'], 3), 'first/last grad', round(v['first_to_last_gradient_ratio'], 4), 'grad by depth', [f'{x:.2e}' for x in v['bank_gradient_norm_by_depth']])


if __name__ == '__main__':
    main()
