"""Consolidate completed B10 future-credit quality, data and work evidence."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
tags={
 'quality':'curie_b10_future_updates_20261013_v2_long',
 'controls':'curie_b10_future_updates_20261013_v1_controls',
 'bptt':'curie_b10_future_updates_20261013_v1_bptt',
 'work':'curie_b10_future_updates_20261013_v1_work',
 'bptt_work':'curie_b10_future_updates_20261013_v1_bptt_work',
 'reuse':'curie_b10_future_credit_reuse_20261013_v1_fit',
 'reuse_updates':'curie_b10_future_credit_reuse_20261013_v1_updates',
 'data_curve':'curie_b10_future_credit_data_20261013_v1_fit',
 'ray':'curie_b10_future_rays_20261013_v1_fit',
 'two_rays':'curie_b10_future_two_rays_20261013_v1_fit'}
paths={key:ROOT/f'experiments/results/market/{tag}.json' for key,tag in tags.items()}
records={key:json.loads(p.read_text()) for key,p in paths.items()}
for r in records.values():
 assert r['status']=='completed'
 for p,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
quality=[*records['quality']['rows'],*records['controls']['rows'],*records['bptt']['rows'],*records['reuse_updates']['rows']]
leader=next(r for r in quality if r['arm']=='learned_refresh')
assert all(leader['nll']<r['nll'] for r in quality if r is not leader)
source=records['quality']['acquisition']
result=dict(status='completed',battle='B10',retained_update='one_pass_learned_refresh',
 quality=[{k:r[k] for k in ('arm','nll','future_nll','targets','updates','causal_refresh_targets')} for r in quality],
 leader_gain={r['arm']:r['nll']-leader['nll'] for r in quality if r is not leader},
 work512=[{k:r[k] for k in ('arm','covered_arithmetic_flops','saved512target_parity_error')} for key in ('work','bptt_work') for r in records[key]['rows']],
 fixed_data_curve=[dict(passes=r['epoch'],unique_credit_train_frames=128,dev_credit_mse=r['dev']['mse']) for r in records['reuse']['curve']],
 half_credit_data_curve=[dict(passes=r['epoch'],unique_credit_train_frames=64,dev_credit_mse=r['dev']['mse']) for r in records['data_curve']['curve']],
 cheap_teacher=[dict(variant=key,dev_credit_mse=records[key]['dev']['mse'],train_teacher_targets=records[key]['train']['forward_teacher_targets']) for key in ('ray','two_rays')],
 original_acquisition_teacher_targets=source['forward_teacher_targets'],
 input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
 source_sha256={'experiments/credit/publish_market_future_credit_evidence.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
 scope='Development quality win: one actor/critic initialization,16384reusedchronologicalDEVtargets,matcheddata/scoring/update schedule. One-pass native credit beats own-carry/refresh,TRAINmean/signed-native and exact8event own-modelBP. Work512is separate covered prefix arithmetic incl127warmup events; no wholefit/energy/matched-compute win. Morecompute is allowed for quality/data efficiency. Credit prediction curves concern additionalcredit data with shared actor prefit: fourpasses128receipts improveMSE but worsenactualactorutility;64receiptsat2passes outperform128at1, laterpassesoverfit. Multipledevelopmentcheckpoints, no sealedTEST/profit/generaldata-efficiency claim.')
with (ROOT/'experiments/results/market/curie_b10_future_credit_evidence_20261013_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
