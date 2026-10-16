"""R1 recursive event depth contracts (no fitting): base equality at init, recursive prediction causality (a unit's mark and
length prediction is unchanged by perturbing tokens inside that unit), stop-gradient at marks, credit to heads and levels."""
import argparse, json, sys, time, hashlib, resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_recursive_hierarchy_lm as rh
import r1_concept_hierarchy_lm as ch
from r1_deep_keyed_lm import DeepKeyedLM
from r1_token_keyed_lm import load


def head_logits(model, x, k, unit):
    model.states(x); low = model.last_starts[k - 1][0]; p = torch.arange(low.shape[0])[low]; begin = p[1:-1][unit]
    head = model.recursive[str(k)]; h = head.n(model.last_levels[k][0, begin - 1]); return head.mark(h), head.gap(h), int(begin), int(p[2:][unit])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--lane', type=int, default=256); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    tr = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); seg = np.asarray(tr[5_000_000:6_000_000]).astype(np.int64); nl = np.isin(seg, (198, 628))
    o = next(i for i in range(0, len(seg) - a.lane - 1, 64) if (seg[i:i + a.lane] == 50256).sum() >= 2 and nl[i:i + a.lane].sum() >= 4)
    x = torch.from_numpy(seg[o:o + a.lane + 1])[None]; rows = list(range(a.lane)); out, checks = dict(window_offset=5_000_000 + o), {}
    torch.manual_seed(0); base = DeepKeyedLM(64, 32, 2, 4, 16, 0.1); torch.manual_seed(0); model = rh.RecursiveHierarchyLM(64, 32, 2, 4, 16, 0.1)
    model.load_state_dict(base.state_dict(), strict=False); base.eval(); model.eval()
    with torch.no_grad(): checks['init_equals_base'] = bool(torch.equal(base.logits(*base.states(x[:, :-1]), x[:, :-1], rows), model.logits(*model.states(x[:, :-1]), x[:, :-1], rows)))
    out['heads'] = sorted(model.recursive.keys())
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if n.startswith(('level_in', 'mix.2')) or n.endswith('.values'): prm.normal_(std=.05)
    m64 = model.double(); errs = []
    with torch.no_grad():
        for k in (int(h) for h in out['heads']):
            low = None
            for unit in (0, 1, 2):
                try: lm, lg, begin, end = head_logits(m64, x[:, :-1], k, unit)
                except IndexError: continue
                if end - begin < 2: continue
                x2 = x.clone(); j = (begin + end) // 2; x2[0, j] = (x2[0, j] + 777) % 50257          # token inside the predicted unit
                st2 = rh.ch.boundaries5(x2[:, :-1])[k - 1][0]
                if not torch.equal(st2[:end + 1], m64.last_starts[k - 1][0][:end + 1]): continue     # perturbation must keep the unit's boundaries
                lm2, lg2, b2, e2 = head_logits(m64, x2[:, :-1], k, unit); errs.append(float(max((lm - lm2).abs().max(), (lg - lg2).abs().max())))
    model.float(); out['recursive_causal_max_change_float64'] = max(errs) if errs else None; out['causal_cases'] = len(errs)
    checks['recursive_causal'] = bool(errs) and max(errs) < 1e-12
    model.train(); h, e = model.states(x[:, :-1]); rl, used = model.recursive_loss(); out['heads_used'] = used
    checks['marks_carry_no_gradient'] = all(not v.requires_grad for v in model.concept_usage.values())
    loss = F.cross_entropy(model.logits(h, e, x[:, :-1], rows).reshape(-1, rh.V), x[:, 1:].reshape(-1)) + 0.1*model.future_loss(x) + 0.1*rl; loss.backward()
    dead = [n for n, prm in model.named_parameters() if n.startswith(('recursive', 'levels', 'concepts')) and (prm.grad is None or float(prm.grad.abs().max()) == 0)]
    out['parameters_without_gradient'] = dead; checks['credit'] = not dead and used == len(out['heads'])
    rec = dict(status='completed' if all(checks.values()) else 'failed', battle='R1', checks=checks, measurements=out,
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/recursive_hierarchy_contracts.py', 'experiments/r1_recursive_hierarchy_lm.py', 'experiments/r1_concept_hierarchy_lm.py', 'experiments/r1_part_whole_lm.py')},
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began)
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(rec, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=rec['status'], checks=checks, **out)))


if __name__ == '__main__':
    main()
