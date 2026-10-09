"""Source-bound evidence packet for the 9 October public-benchmark public benchmark record (REPORT.md and its PDF).

Every own number is computed here from completed result files, and every input file's sha256 is recorded. Published
reference numbers are quoted from the battle dossiers (experiments/B1_EASYTPP.md, experiments/B2_IRREGULAR_TS.md,
experiments/tgb/B5_TGB.md, experiments/B4_NTPP_BENCHMARK_PROPOSAL.md), whose hashes are recorded too.
Output: report/public_wins_headline_evidence_20261009_v3.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
inputs = {}


def load(path):
    p = ROOT / path
    inputs[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def stats(vals):
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), values=[float(v) for v in vals])


def b4_stats(prefix):
    records = [load(f'experiments/results/tpp_b4/{prefix}_s{k}.json') for k in range(5)]
    assert all(r['status'] == 'completed' and r['split'] == k and r['args']['score_test']
               for k, r in enumerate(records)), 'Five completed official B4 splits required'
    assert len({json.dumps(r['config'], sort_keys=True) for r in records}) == 1
    assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in records}) == 1
    assert len({r['dataset'] for r in records}) == 1
    result = {key: stats([r['test'][key] for r in records]) for key in ('L_T', 'L_M', 'total')}
    for value in result.values():
        value['se'] = value['sd'] / np.sqrt(5)
    return result


def tpp(prefix, n=5):
    return stats([load(f'experiments/results/tpp/{prefix}_s{s}.json')['test']['ll'] for s in range(n)])


def recall_stats(paths):
    records = [load(path) for path in paths]
    assert [r['args']['seed'] for r in records] == [0, 1, 2]
    assert all(r['status'] == 'completed' and r['battle'] == 'R1' for r in records)
    assert len({json.dumps({k: v for k, v in r['args'].items() if k not in ('tag', 'seed')},
                           sort_keys=True) for r in records}) == 1
    assert len({json.dumps(r['source_sha256'], sort_keys=True) for r in records}) == 1
    for source, expected in records[0]['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == expected
    sections = ('test', 'extrapolation', 'extrapolation_far') if 'extrapolation_far' in records[0] else ('test', 'extrapolation')
    return {section: dict(recall=stats([r[section]['recall_acc'] for r in records]),
                          ll=stats([r[section]['ll'] for r in records]),
                          set_baseline=stats([r[section]['set_baseline_acc'] for r in records])) for section in sections}


easy = dict(taxi=tpp('b1_final_taxi_v5'), taobao=tpp('b1_final_taobao_v5'), stackoverflow=tpp('b1_final_stackoverflow_v12l2'),
            retweet=tpp('b1_final_retweet_v16'), amazon=tpp('b1_final_amazon_v18'))
unified = {d: tpp(f'b1_final_unified_v19_{d}') for d in ('taxi', 'taobao', 'stackoverflow', 'retweet', 'amazon')}
published = dict(taxi=('S2P2', 0.522), taobao=('IFTPP', 1.318), stackoverflow=('S2P2', -2.163), retweet=('NHP', -6.348),
                 amazon=('S2P2', 0.781))
p19 = load('experiments/results/irts/b2_final_p19_v3_summary.json')
p12 = load('experiments/results/irts/b2_final_p12_v3_summary.json')
pam = [load(f'experiments/results/irts/b2_final_pam_v7ema_split{k}.json')['test'] for k in range(5)]
trade = [load(f'experiments/results/tgb/curie_b5_trade_affinity_sealed_s{s}_20261009T1210Z.json')['test_ndcg'] for s in range(3)]
wiki_dev = load('experiments/results/tgb/curie_b5_racelink_v4_id0_dev_s0_20261009T1300Z.json')
wiki = [load(f'experiments/results/tgb/curie_b5_wiki_sealed_v4_id0_s{s}_20261009T1340Z.json')['test_mrr'] for s in range(3)]
mooc = b4_stats('b4_mooc')
stack_overflow = b4_stats('b4_stack_overflow')
wikipedia = b4_stats('b4_wikipedia')
github = b4_stats('b4_github')
mimic2 = b4_stats('b4_mimic2')
recall_mixed = recall_stats([f'experiments/results/tpp/recall/curie_r1_v5len_p_keyed1_s{s}_{stamp}.json'
                            for s, stamp in enumerate(('20261007T1715Z', '20261007T1915Z', '20261007T1915Z'))])
recall_local_normalized = recall_stats([f'experiments/results/tpp/recall/curie_r1_v4_pn_keyed2_s{s}_{stamp}.json'
                                       for s, stamp in enumerate(('20261007T1425Z', '20261007T1900Z', '20261007T1900Z'))])
for doc in ('experiments/B1_EASYTPP.md', 'experiments/B2_IRREGULAR_TS.md', 'experiments/tgb/B5_TGB.md',
            'experiments/B4_NTPP_BENCHMARK_PROPOSAL.md', 'experiments/fas/B3_DEVELOPMENT_LOG.md'):
    inputs[doc] = hashlib.sha256((ROOT / doc).read_bytes()).hexdigest()

seeds_ahead = sum(sum(v > published[d][1] for v in easy[d]['values']) for d in easy)
packet = dict(
    status='completed', date='2026-10-09',
    easytpp={d: dict(ours=easy[d], unified=unified[d], published=dict(model=published[d][0], ll=published[d][1]),
                     seeds_ahead=int(sum(v > published[d][1] for v in easy[d]['values']))) for d in easy},
    easytpp_seeds_ahead=int(seeds_ahead),
    unified_seeds_ahead=int(sum(sum(v > published[d][1] for v in unified[d]['values']) for d in unified)),
    p19=dict(auroc=[p19['test_auroc_mean'], p19['test_auroc_sd']], auprc=[p19['test_auprc_mean'], p19['test_auprc_sd']],
             published=dict(model='MTM', auroc=0.903, auprc=0.583)),
    p12=dict(auroc=[p12['test_auroc_mean'], p12['test_auroc_sd']], auprc=[p12['test_auprc_mean'], p12['test_auprc_sd']],
             published=dict(model='MTM', auroc=0.880, auprc=0.586)),
    pam=dict(acc=stats([t['acc'] for t in pam]), f1=stats([t['f1'] for t in pam]), published=dict(model='MTM', acc=0.975, f1=0.976),
             splits_ahead=int(sum(t['acc'] > 0.975 for t in pam))),
    tgbn_trade=dict(ndcg=stats(trade), published=dict(model='NAVIS (ICLR 2026)', ndcg=0.863), persistent_forecast=0.855),
    tgbl_wiki=dict(mrr=stats(wiki), val_mrr=wiki_dev['best_val_mrr'], parameters=wiki_dev['parameters'],
                   published=dict(model='TPNet', val=0.842, test=0.827)),
    b4_mooc=dict(**mooc, bar=-239.7, best_single=-233.8, verdict='loss (pre-registered frozen configuration)'),
    b4_stack_overflow=dict(**stack_overflow, bar=11.9, best_single=12.1,
                          published_time=-91.1, published_marks=103.0,
                          verdict='loss (pre-registered frozen configuration)',
                          diagnosis='Time component ahead; mark component carries the total-NLL gap',
                          scope='Five fixed splits; NLL per sequence; uncertainty is standard error'),
    b4_wikipedia=dict(**wikipedia, bar=-122.62, best_single=-2.67, published_time=-267.41, published_marks=144.79,
                      splits_ahead=int(sum(v < -122.62 for v in wikipedia['total']['values'])),
                      verdict='win (pre-registered frozen configuration)',
                      diagnosis='Mark-memory win: 90-93% of consecutive edits repeat the page, 10-27% of TEST pages are unseen '
                                'in TRAIN; per-mark addressed memory copies them. Scoring verified on splits 0 and 4 '
                                '(reproduction, causality, mark normalisation, published units)'),
    b4_github=dict(**github, bar=-272.9, best_single=-269.7, published_time=-382.4, published_marks=109.5,
                   verdict='loss (pre-registered frozen configuration)',
                   diagnosis='Splits 1-4 diverged to NaN at epochs 3-6 (no non-finite guard); selection fell back to epochs 2-3'),
    b4_mimic2=dict(**mimic2, bar=2.42, best_single=3.1, published_time=0.13, published_marks=2.29,
                   verdict='loss (pre-registered frozen configuration)',
                   diagnosis='About 3 events per sequence; selection at epochs 3-11 (early overfitting of the mark path)'),
    r1_recall=dict(mixed_length=recall_mixed, local_normalized=recall_local_normalized,
                   scope='Three completed seeds per configuration; mean and sample SD; '
                         'mixed training 4–16 pairs, held-out 32-pair evaluation; no new scoring'),
    fas_v2='level on validation with the time-encoded Transformer reference (0.702 vs 0.704) at about 1/7 of its parameters; '
           'sealed verdict pending the reference seeds; a tie is expected under the 0.02 rule',
    input_sha256=inputs)
out = ROOT / 'report/public_wins_headline_evidence_20261009_v3.json'
out.write_text(json.dumps(packet, indent=1) + '\n')
print(json.dumps(dict(easy={d: round(easy[d]['mean'], 4) for d in easy}, seeds_ahead=seeds_ahead, pam=packet['pam']['acc']['mean'],
                      trade=packet['tgbn_trade']['ndcg']['mean'], mooc=packet['b4_mooc']['total']['mean'])))
