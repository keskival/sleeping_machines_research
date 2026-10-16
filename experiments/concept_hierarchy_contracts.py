"""R1 concept hierarchy contracts (no fitting): base equality at init, causality, credit to levels and concept memories,
observed level rates on real tokens, inference memory/time on the real scoring shapes."""
import argparse, json, sys, time, hashlib, resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_concept_hierarchy_lm as ch
import r1_part_whole_lm as pw
from r1_deep_keyed_lm import DeepKeyedLM
from r1_token_keyed_lm import load, evaluate


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--scoring-shapes', type=int, default=1); p.add_argument('--tree', type=int, default=0); p.add_argument('--lane', type=int, default=1024); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    tr = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); seg = np.asarray(tr[5_000_000:6_000_000]).astype(np.int64); nl = np.isin(seg, (198, 628))
    o = next(i for i in range(0, len(seg) - a.lane - 1, 64) if (seg[i:i + a.lane] == 50256).sum() >= 2 and nl[i:i + a.lane].sum() >= 4)   # window containing the structure under test
    x = torch.from_numpy(seg[o:o + a.lane + 1])[None]; rows = list(range(a.lane)); window_offset = 5_000_000 + o
    out, checks = dict(window_offset=window_offset), {}
    torch.manual_seed(0); base = DeepKeyedLM(64, 32, 2, 4, 16, 0.1); torch.manual_seed(0); model = ch.ConceptHierarchyLM(64, 32, 2, 4, 16, 0.1, tree=bool(a.tree))
    model.load_state_dict(base.state_dict(), strict=False); base.eval(); model.eval()
    with torch.no_grad():
        lb = base.logits(*base.states(x[:, :-1]), x[:, :-1], rows); lm = model.logits(*model.states(x[:, :-1]), x[:, :-1], rows)
    checks['init_equals_base'] = bool(torch.equal(lb, lm))
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if n.startswith(('level_in', 'mix.2', 'tree.o')) or n.endswith('.values'): prm.normal_(std=.05)
        # Causality in float64: data-dependent entry counts change matrix shapes, hence summation order (rounding ~1e-15);
        # a leak would move earlier logits by orders of magnitude more.
        cut = a.lane // 2; x2 = x.clone(); x2[:, cut] = (x2[:, cut] + 777) % 50257; m64 = model.double()
        l1 = m64.logits(*m64.states(x[:, :-1]), x[:, :-1], rows); l2 = m64.logits(*m64.states(x2[:, :-1]), x2[:, :-1], rows); model.float()
    out['prefix_max_change_float64'] = float((l1[:, :cut] - l2[:, :cut]).abs().max()); out['suffix_max_change_float64'] = float((l1[:, cut:] - l2[:, cut:]).abs().max())
    checks['causal'] = out['prefix_max_change_float64'] < 1e-12 and out['suffix_max_change_float64'] > 1e-3
    if a.tree:   # tree weights are a distribution: token mass + null mass = 1 for every query
        with torch.no_grad():
            model.tree.null.fill_(-30.); model.states(x[:, :-1]); out['tree_leaf_mass_error'] = float((model.tree.last_mass - 1).abs().max())
            model.tree.null.fill_(0.); model.states(x[:, :-1]); out['tree_leaf_mass_range_null0'] = [float(model.tree.last_mass.min()), float(model.tree.last_mass.max())]
        checks['tree_distribution'] = out['tree_leaf_mass_error'] < 1e-5
    model.train(); h, e = model.states(x[:, :-1]); loss = F.cross_entropy(model.logits(h, e, x[:, :-1], rows).reshape(-1, pw.V), x[:, 1:].reshape(-1)) + 0.1*model.future_loss(x); loss.backward()
    dead = [n for n, prm in model.named_parameters() if n.startswith(('levels', 'concepts', 'tree')) and (prm.grad is None or float(prm.grad.abs().max()) == 0)]
    out['parameters_without_gradient'] = dead; checks['credit'] = not dead
    big = torch.from_numpy(np.asarray(tr[6_000_000:6_000_000 + 1024*1024]).astype(np.int64)).reshape(1024, 1024)
    rates = [float(s.double().mean()) for s in ch.boundaries5(big)]; out['span_start_rate'] = dict(zip(('word', 'clause', 'sentence', 'line', 'document'), rates))
    out['mean_span_tokens'] = {k: 1/v for k, v in out['span_start_rate'].items()}; checks['nested_rates'] = all(r1 >= r2 for r1, r2 in zip(rates, rates[1:]))
    if a.scoring_shapes:
        model.eval(); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin'); shapes = {}
        for name, fn in (('select_8x2048', lambda: evaluate(model, val, 10485760, 8, 2048)), ('one_8192_lane', lambda: evaluate(model, val, 20971520, 1, 8192)), ('flagship_8x1024', lambda: pw.flagship(model, val))):
            t0 = time.time(); fn(); shapes[name] = dict(wall_s=time.time() - t0, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out['scoring_shapes'] = shapes; checks['scoring_memory_within_job_cap'] = max(v['peak_rss_kb'] for v in shapes.values()) < 2_500_000
    rec = dict(status='completed' if all(checks.values()) else 'failed', battle='R1', checks=checks, measurements=out,
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/concept_hierarchy_contracts.py', 'experiments/r1_concept_hierarchy_lm.py', 'experiments/r1_part_whole_lm.py', 'experiments/r1_deep_keyed_lm.py')},
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began)
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(rec, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=rec['status'], checks=checks, **{k: out[k] for k in ('prefix_max_change_float64', 'parameters_without_gradient', 'mean_span_tokens')}, shapes=out.get('scoring_shapes'))))


if __name__ == '__main__':
    main()
