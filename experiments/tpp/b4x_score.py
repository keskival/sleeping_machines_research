#!/usr/bin/env python3
"""B4 developed attempt: once-only TEST scoring. For one dataset, choose the variant with the lowest mean validation
total over the five splits (no TEST information), load that variant's saved per-split checkpoints, score each TEST split
once, and compare the five-split mean with the preregistered composite target and the best single published model."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_b4x as X  # noqa: E402
from race_tpp_b4g import load, per_sequence  # noqa: E402

TARGET = {'wikipedia_filtered': -122.62, 'github_filtered': -272.90, 'stack_overflow_filtered': 11.9, 'lastfm_filtered': -849.65,
          'mooc_filtered': -239.7, 'retweets_filtered': -538.70, 'mimic2_filtered': None}
BEST_SINGLE = {'wikipedia_filtered': -2.67, 'github_filtered': -269.7, 'stack_overflow_filtered': 12.1, 'lastfm_filtered': -697.5,
               'mooc_filtered': -233.8, 'retweets_filtered': -536.17, 'mimic2_filtered': 3.1}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--variants', required=True, help='comma list of result tag prefixes; each has _s{split} results'); a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); out = ROOT/f'experiments/results/tpp_b4/{a.tag}.json'; assert not out.exists()
    rows = {}
    for v in a.variants.split(','):
        res = [json.loads((ROOT/f'experiments/results/tpp_b4/{v}_s{s}.json').read_text()) for s in range(5)]
        assert all(r['status'] == 'completed' and r['dataset'] == a.dataset for r in res)
        rows[v] = dict(val_total_mean=float(np.mean([r['val']['total'] for r in res])), results=res)
    chosen = min(rows, key=lambda v: rows[v]['val_total_mean']); test = []
    for s, r in enumerate(rows[chosen]['results']):
        model, _, _, _ = X.build(a.dataset, s, r['args']['seed'], r['args'].get('pointer', False)); model.load_state_dict(torch.load(ROOT/r['checkpoint'])); model.eval()
        test.append(per_sequence(model, load(a.dataset, s, 'test')[0]))
    mean = {k: float(np.mean([x[k] for x in test])) for k in ('L_T', 'L_M', 'total')}; se = float(np.std([x['total'] for x in test], ddof=1)/np.sqrt(5))
    tgt, single = TARGET[a.dataset], BEST_SINGLE[a.dataset]
    record = dict(status='completed', battle='B4', attempt='developed', dataset=a.dataset, chosen_variant=chosen, selection='lowest five-split mean validation total; TEST unused',
                  val_means={v: rows[v]['val_total_mean'] for v in rows}, test_splits=test, test_mean=mean, test_total_se=se,
                  composite_target=tgt, beats_composite_target=(mean['total'] < tgt) if tgt is not None else None, best_single_published=single, beats_best_single=mean['total'] < single,
                  checkpoints_sha256={r['checkpoint']: sha(ROOT/r['checkpoint']) for r in rows[chosen]['results']},
                  source_sha256={n: sha(ROOT/n) for n in ['experiments/tpp/b4x_score.py'] + X.SOURCES}, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    out.write_text(json.dumps(record, indent=1) + '\n'); print(json.dumps({k: record[k] for k in ('dataset', 'chosen_variant', 'test_mean', 'composite_target', 'beats_composite_target', 'beats_best_single')}))


if __name__ == '__main__':
    main()
