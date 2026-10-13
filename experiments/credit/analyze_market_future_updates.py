"""Current source-bound actual-update trajectory/work comparison."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
files={name:ROOT/f'experiments/results/market/{tag}.json' for name,tag in (
    ('short','curie_b10_future_updates_20261013_v1_fit'),
    ('long','curie_b10_future_updates_20261013_v2_long'),
    ('work','curie_b10_future_updates_20261013_v1_work'),
    ('controls','curie_b10_future_updates_20261013_v1_controls'))}
records={key:json.loads(path.read_text()) for key,path in files.items()}
for r in records.values():
    for name,digest in r['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
by={r['arm']:r for r in records['long']['rows']};prefix={}
for old in records['short']['rows']:
    new=by[old['arm']]
    error=max(abs(a[key]-b[key]) for a,b in zip(old['rows'],new['rows'][:128]) for key in ('producer_nll','future_nll'))
    assert error==0;prefix[old['arm']]=error
learned=by['learned_refresh'];matched=by['own_refresh']
controls=records['controls']['rows']
windows=[]
for indexes in np.array_split(np.arange(len(learned['rows'])),4):
    def mean(arm):return float(np.mean([.5*(arm['rows'][i]['producer_nll']+arm['rows'][i]['future_nll']) for i in indexes]))
    windows.append(dict(start_frame=int(indexes[0]),end_frame=int(indexes[-1])+1,
                        learned_nll=mean(learned),matched_nll=mean(matched),matched_minus_learned=mean(matched)-mean(learned)))
result=dict(status='completed',battle='B10',quality=[{key:row[key] for key in ('arm','targets','updates','nll','future_nll','causal_refresh_targets','wall_s')} for row in by.values()],
    learned_gain_vs_matched=matched['nll']-learned['nll'],
    learned_future_gain_vs_matched=matched['future_nll']-learned['future_nll'],
    additional_controls=[{key:r[key] for key in ('arm','nll','future_nll','causal_refresh_targets','wall_s')} for r in controls],
    learned_gains_vs_controls={r['arm']:r['nll']-learned['nll'] for r in controls},
    prefix_exact_error=prefix,chronological_quarters=windows,
    work512=[{key:row[key] for key in ('arm','covered_arithmetic_flops','saved512target_parity_error')} for row in records['work']['rows']],
    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
    source_sha256={'experiments/credit/analyze_market_future_updates.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    scope='One acquired critic/actor initialization, reused16384chronological DEVtargets, new producer8/future8 protocol. Compared against matched own-refresh/carry controls; source and512work additionally paid. Work512includes warmup/declared covered kernels, is not a full16384measurement or acquisition/energy total. Current market32target headline unchanged; no sealedTEST/public benchmark/profit claim.')
with (ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
# Publish a portable numerical state packet: binary checkpoints remain preserved
# locally but are ignored by git. JSON floats round-trip the acquired weights.
checkpoint=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v2_long_critic.pt'
packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
weights={name:dict(dtype=str(t.dtype),shape=list(t.shape),values=t.flatten().tolist()) for name,t in packet['state_dict'].items()}
for name,row in weights.items():
    reconstructed=torch.tensor(row['values'],dtype=packet['state_dict'][name].dtype).reshape(row['shape'])
    assert torch.equal(reconstructed,packet['state_dict'][name]),name
bundle=dict(status='completed',battle='B10',weights=weights,scale=packet['scale'],
            binary_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            acquisition_parent_sha256=hashlib.sha256(files['long'].read_bytes()).hexdigest(),
            scope='Frozen acquired native width264/four-coordinate critic, state_dict numerical round-trip exact; reset private episode state for restoration. No actor weights or predictor acquisition omitted from costing.')
with (ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_critic_state.json').open('x') as f:json.dump(bundle,f);f.write('\n')
print(json.dumps(result),flush=True)
