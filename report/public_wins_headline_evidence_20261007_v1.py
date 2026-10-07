"""Source-bound evidence packet for the 7 October public-benchmark headline page of the status report.

Every own number is computed here from completed result files, and every input file's sha256 is recorded. Published
reference numbers and per-event compute ratios are quoted from the battle dossiers (experiments/B1_EASYTPP.md,
experiments/B2_IRREGULAR_TS.md), whose hashes are recorded too.
Output: report/public_wins_headline_evidence_20261007_v1.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'experiments/results'
inputs = {}


def load(path):
    p = ROOT / path
    inputs[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def seeds(prefix):
    lls = [load(f'experiments/results/tpp/{prefix}_s{s}.json')['test']['ll'] for s in range(5)]
    return dict(mean=float(np.mean(lls)), sd=float(np.std(lls, ddof=1)), seeds=lls)


taxi = seeds('b1_final_taxi_v5'); taxi['mixture'] = load('experiments/results/tpp/b1_final_taxi_v5_ensemble5_test.json')['ensemble']['ll']
taobao = seeds('b1_final_taobao_v5')
so_matched = seeds('b1_final_stackoverflow_v12l2')
so_first = seeds('b1_final_stackoverflow_v12')
amazon = load('experiments/results/tpp/b1_final_amazon_v11r_selected.json')
p19 = load('experiments/results/irts/b2_final_p19_v3_summary.json')
p12 = load('experiments/results/irts/b2_final_p12_v3_summary.json')
for doc in ('experiments/B1_EASYTPP.md', 'experiments/B2_IRREGULAR_TS.md', 'experiments/fas/B3_DEVELOPMENT_LOG.md'):
    inputs[doc] = hashlib.sha256((ROOT / doc).read_bytes()).hexdigest()

packet = dict(
    status='completed', date='2026-10-07',
    easytpp=dict(
        taxi=dict(ours=taxi, published=dict(model='S2P2 (NeurIPS 2025)', ll=0.522, sd=0.004),
                  compute='single model 1/12 of S2P2 parameters and per-event MACs; 5-seed mixture 0.41x', verdict='WIN, confirmed (5 seeds)'),
        taobao=dict(ours=taobao, published=dict(model='IFTPP', ll=1.318, sd=0.017), compute='0.92x S2P2 per-event MACs',
                    verdict='WIN, confirmed (5 seeds)'),
        stackoverflow=dict(ours_matched=so_matched, ours_first=so_first, published=dict(model='S2P2', ll=-2.163, sd=0.009),
                           compute='matched size: 0.968x parameters, 1.015x per-event MACs (first win at 1.26x)',
                           verdict='WIN at matched compute, confirmed (5 seeds)'),
        amazon=dict(ours=dict(mean=amazon['test_ll_mean'], sd=amazon['test_ll_sd']), published=dict(model='S2P2', ll=0.781),
                    compute='0.29x S2P2 per-event MACs', verdict='mean ahead, not confirmed (one low-basin seed)'),
        retweet=dict(verdict='in TEST protocol; earlier leads withdrawn by the recording-grid audit')),
    clinical=dict(
        p19=dict(auroc=[p19['test_auroc_mean'], p19['test_auroc_sd']], auprc=[p19['test_auprc_mean'], p19['test_auprc_sd']],
                 published=dict(model='MTM', auroc=[0.903, 0.020], auprc=[0.583, 0.053]),
                 verdict='WIN (five official splits): AUPRC beyond both split spreads, AUROC within one split sd'),
        p12=dict(auroc=[p12['test_auroc_mean'], p12['test_auroc_sd']], auprc=[p12['test_auprc_mean'], p12['test_auprc_sd']],
                 published=dict(model='MTM', auroc=0.880, auprc=0.586),
                 verdict='behind MTM on AUROC, level on AUPRC; ahead of all other published models')),
    fas_v2='in development: best v2 validation AUROC .663 (C2) vs best anonymous classical .685; round 2 (recording-cell likelihood) running',
    input_sha256=inputs)
out = ROOT / 'report/public_wins_headline_evidence_20261007_v1.json'
out.write_text(json.dumps(packet, indent=1) + '\n')
print(json.dumps(dict(taxi=taxi['mean'], mixture=taxi['mixture'], taobao=taobao['mean'], so=so_matched['mean'],
                      amazon=amazon['test_ll_mean'], p19=p19['test_auprc_mean'])))
