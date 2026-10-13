"""Three independently initialized native critics on the fixed B10 protocol."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
parent=ROOT/'experiments/results/market/curie_b10_future_credit_evidence_20261013_v1.json'
base=json.loads(parent.read_text());inputs={str(parent.relative_to(ROOT)):hashlib.sha256(parent.read_bytes()).hexdigest()}
first=next(r for r in base['quality'] if r['arm']=='learned_refresh')
rows=[dict(critic_seed=19124,nll=first['nll'],future_nll=first['future_nll'])]
for seed in (19129,19130):
 p=ROOT/f'experiments/results/market/curie_b10_future_credit_confirm_20261013_s{seed}_v1.json'
 r=json.loads(p.read_text());assert r['status']=='completed'
 for name,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 rows.append(dict(critic_seed=seed,nll=r['rows'][0]['nll'],future_nll=r['rows'][0]['future_nll'],dev_credit_mse=r['dev']['mse']))
controls={r['arm']:r['nll'] for r in base['quality'] if r['arm'] in ('own_carry','own_refresh','bptt_carry','bptt_refresh','train_mean_refresh')}
nll=np.array([r['nll'] for r in rows]);gains={arm:value-nll for arm,value in controls.items()}
result=dict(status='completed',battle='B10',rows=rows,mean_nll=float(nll.mean()),sample_sd_nll=float(nll.std(ddof=1)),
 mean_gain={arm:float(g.mean()) for arm,g in gains.items()},all_seed_gains={arm:g.tolist() for arm,g in gains.items()},
 lower_than_control_counts={arm:int((g>0).sum()) for arm,g in gains.items()},input_sha256=inputs,
 source_sha256={'experiments/credit/publish_market_future_credit_confirmation.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
 scope='Three independently initialized nativecriticfits, shared fixed restoredactor/common128TRAINforwardreceipts and16384reusedDEVevents; actor updates online under identical8+8/clippedAdam/refresh schedule. Nativeweightsfrozen duringDEV, privatehistorycausal. SampleSD across criticinitializations, not actor/datareplicates. Deterministic own/BP/TRAINmean controls reused; originalsignednativecontrol belongs to seed19124only. Extra compute allowed for quality; source/data/512work separate, no sealedTEST/profit/fullmodeldata-efficiency/energy claim.')
with (ROOT/'experiments/results/market/curie_b10_future_credit_confirmation_20261013_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
