"""R1 contracts for per-stage addressed reads on the residual race stack (no fitting)."""
import argparse, json, sys, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch
import multidepth_readout_token_language as serial_driver
import residual_depth_token_language as driver
from residual_depth_contracts import config


def cfg(depth, residual, reads):
    a = config(depth, residual); a.stage_reads = reads; a.read_key_dim = 16; a.read_topk = 16; a.read_capacity = 512
    return a


def run(m, a, ids):
    g = torch.Generator().manual_seed(1); x, s, _ = m(ids[:, :-1], None, g, a); return x, s


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a0 = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    base = cfg(2, False, False); train = serial_driver.load_tokens(base.train_file)
    counts = serial_driver.LaneTokens(train, 0, base.train_tokens, base.lanes).counts(); order = torch.argsort(counts, descending=True, stable=True)
    ids = torch.from_numpy(train[3_000_000:3_000_000 + 8 * 65].astype('int64')).reshape(8, 65)
    out = {}
    # (b) no reads, no residual: original serial driver bit for bit.
    a = cfg(4, False, False); torch.manual_seed(6); m1 = serial_driver.Model(a, order, counts); torch.manual_seed(6); m2 = driver.Model(a, order, counts); m2.load_state_dict(m1.state_dict())
    x1, _ = run(m1, a, ids); x2, _ = run(m2, a, ids); out['serial_equal'] = bool(torch.equal(x1, x2))
    # (a) zero value map: read stack equals residual stack bit for bit.
    ar, arr_ = cfg(4, True, False), cfg(4, True, True)
    torch.manual_seed(6); mr = driver.Model(ar, order, counts); torch.manual_seed(6); mrr = driver.Model(arr_, order, counts)
    mrr.load_state_dict(mr.state_dict(), strict=False)
    xr, _ = run(mr, ar, ids); xrr, srr = run(mrr, arr_, ids); out['zero_value_equal_residual'] = bool(torch.equal(xr, xrr))
    out['store_entries_after_64_tokens'] = int(srr['store_k'][0].shape[1])
    # (c) read-time credit with a nonzero value map, two credit windows (chunk 64, window 16).
    with torch.no_grad(): mrr.core.read_value.normal_(std=.05); mrr.core.read_boundary.fill_(-.5)
    idsb = ids.clone(); idsb[:, 20] = 50256  # one document boundary per lane
    mrr.zero_grad(); xg, _ = run(mrr, arr_, idsb); (xg.square().mean()).backward()
    out['grad_norms'] = {n: float(getattr(mrr.core, n).grad.norm()) for n in ('read_query', 'read_key', 'read_value', 'read_age', 'read_boundary', 'read_null')}
    # (d) causality: change token 40; features before position 40 unchanged, later ones change.
    ids2 = ids.clone(); ids2[:, 40] = (ids2[:, 40] + 101) % 50257
    with torch.no_grad():
        xa, _ = run(mrr, arr_, ids); xb, _ = run(mrr, arr_, ids2)
    out['causal_prefix_max_change'] = float((xa[:, :40] - xb[:, :40]).abs().max()); out['suffix_changes'] = bool((xa[:, 40:] - xb[:, 40:]).abs().max() > 0)
    checks = dict(serial_equal=out['serial_equal'], zero_value_equal_residual=out['zero_value_equal_residual'],
                  credit=all(v > 0 for v in out['grad_norms'].values()), causal=out['causal_prefix_max_change'] == 0 and out['suffix_changes'],
                  store=out['store_entries_after_64_tokens'] == 64)
    rec = dict(status='completed' if all(checks.values()) else 'failed', battle='R1', checks=checks, measurements=out,
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('experiments/residual_read_contracts.py', 'experiments/residual_depth_token_language.py', 'sleeping_machines/residual_race_episodes.py')},
               process_wall_s=time.monotonic() - began)
    with (ROOT / f'experiments/results/diagnostics/{a0.tag}.json').open('x') as f: json.dump(rec, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=rec['status'], checks=checks, grads=out['grad_norms'])))


if __name__ == '__main__':
    main()
