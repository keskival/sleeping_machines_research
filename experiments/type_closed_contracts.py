"""R1 type-closed additive stack contracts (no fitting)."""
import argparse, json, sys, time, hashlib, resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_type_closed_lm as tc
from r1_deep_keyed_lm import DeepKeyedLM
from r1_token_keyed_lm import load


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--lane', type=int, default=256); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    tr = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); seg = np.asarray(tr[5_000_000:6_000_000]).astype(np.int64); nl = np.isin(seg, (198, 628))
    o = next(i for i in range(0, len(seg) - a.lane - 1, 64) if (seg[i:i + a.lane] == 50256).sum() >= 2 and nl[i:i + a.lane].sum() >= 4)
    x = torch.from_numpy(seg[o:o + a.lane + 1])[None]; rows = list(range(a.lane)); out, checks = dict(window_offset=5_000_000 + o), {}
    torch.manual_seed(0); base = DeepKeyedLM(64, 32, 2, 4, 16, 0.1); torch.manual_seed(0); model = tc.TypeClosedLM(64, 32, 2, 4, 16, 0.1)
    model.load_state_dict(base.state_dict(), strict=False); base.eval(); model.eval()
    with torch.no_grad():
        checks['init_equals_base'] = bool(torch.equal(base.logits(*base.states(x[:, :-1]), x[:, :-1], rows), model.logits(*model.states(x[:, :-1]), x[:, :-1], rows)))
        errs = []
        for z in model.cache:
            lv, c, w = z['level'], z['c'], z['whole']; inp = lv.lexicon[c] + lv.P(w - lv.codebook.keys[c]); errs.append(float(((inp - w) * z['real'][..., None]).abs().max()))
        out['interface_identity_max_error'] = max(errs); checks['interface_identity_at_init'] = max(errs) < 1e-5
        out['events_per_level'] = [int(z['real'].sum()) for z in model.cache]
        lv = model.levels[0]; p1, p2 = torch.randn(1, 40, 64, dtype=torch.float64), torch.randn(1, 40, 64, dtype=torch.float64); us = torch.zeros(1, 40, dtype=torch.bool); us[0, [0, 7, 19, 30]] = True
        lv64 = lv.double(); f = lambda q: lv64.part_out(tc.pw.span_integrate(lv64.part, q, us.long().cumsum(1), torch.ones_like(us)))
        out['whole_linearity_error'] = float((f(2.0 * p1 - 3.0 * p2) - (2.0 * f(p1) - 3.0 * f(p2))).abs().max()); lv.float(); checks['linear_part_path'] = out['whole_linearity_error'] < 1e-10
        cb = tc.Codebook(4, 3, decay=0.5); k0, n0, t0 = cb.keys.clone(), cb.count.clone(), cb.total.clone(); w = torch.randn(5, 4); c = torch.tensor([0, 0, 2, 1, 0]); cb.update(w, c)
        n = torch.bincount(c, minlength=3).float(); s = torch.zeros(3, 4).index_add_(0, c, w); tot = 0.5 * t0 + 0.5 * s; cnt = 0.5 * n0 + 0.5 * n
        out['codebook_recurrence_error'] = float((cb.keys - tot / cnt.unsqueeze(-1)).abs().max()); checks['codebook_recurrence'] = out['codebook_recurrence_error'] < 1e-6
        for n_, prm in model.named_parameters():
            if n_.startswith(('A.', 'levels')) and ('corr.2' in n_ or n_.startswith('A.')): prm.normal_(std=.05)
        m64 = model.double(); cut = a.lane // 2; x2 = x.clone(); x2[0, cut] = (x2[0, cut] + 777) % 50257
        l1 = m64.logits(*m64.states(x[:, :-1]), x[:, :-1], rows); l2 = m64.logits(*m64.states(x2[:, :-1]), x2[:, :-1], rows); model.float()
    out['prefix_max_change_float64'] = float((l1[:, :cut] - l2[:, :cut]).abs().max()); out['suffix_max_change_float64'] = float((l1[:, cut:] - l2[:, cut:]).abs().max())
    checks['causal'] = out['prefix_max_change_float64'] < 1e-12 and out['suffix_max_change_float64'] > 1e-3
    model.train(); h, e = model.states(x[:, :-1]); rl, used = model.recursive_loss()
    (F.cross_entropy(model.logits(h, e, x[:, :-1], rows).reshape(-1, tc.V), x[:, 1:].reshape(-1)) + 0.1 * rl).backward()
    dead = [n_ for n_, prm in model.named_parameters() if n_.startswith(('levels', 'A.')) and prm.requires_grad and (prm.grad is None or float(prm.grad.abs().max()) == 0)]
    out['parameters_without_gradient'] = dead; out['recursive_levels_used'] = used; checks['credit'] = not dead
    rec = dict(status='completed' if all(checks.values()) else 'failed', battle='R1', checks=checks, measurements=out,
               source_sha256={n_: hashlib.sha256((ROOT/n_).read_bytes()).hexdigest() for n_ in ('experiments/type_closed_contracts.py', 'experiments/r1_type_closed_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_part_whole_lm.py')},
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began)
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(rec, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=rec['status'], checks=checks, **out)))


if __name__ == '__main__':
    main()
