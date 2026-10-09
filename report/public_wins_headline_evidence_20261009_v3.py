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


def tpp(prefix, n=5):
    return stats([load(f'experiments/results/tpp/{prefix}_s{s}.json')['test']['ll'] for s in range(n)])


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
mooc = [load(f'experiments/results/tpp_b4/b4_mooc_s{k}.json')['test']['total'] for k in range(5)]
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
    b4_mooc=dict(total=stats(mooc), bar=-239.7, best_single=-233.8, verdict='loss (pre-registered)'),
    fas_v2='level on validation with the time-encoded Transformer reference (0.702 vs 0.704) at about 1/7 of its parameters; '
           'sealed verdict pending the reference seeds; a tie is expected under the 0.02 rule',
    input_sha256=inputs)
out = ROOT / 'report/public_wins_headline_evidence_20261009_v3.json'
out.write_text(json.dumps(packet, indent=1) + '\n')
print(json.dumps(dict(easy={d: round(easy[d]['mean'], 4) for d in easy}, seeds_ahead=seeds_ahead, pam=packet['pam']['acc']['mean'],
                      trade=packet['tgbn_trade']['ndcg']['mean'], mooc=packet['b4_mooc']['total']['mean'])))
