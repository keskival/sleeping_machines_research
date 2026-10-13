"""Source-bound B10 optimizer-conditioned credit cohort summary."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def load(tag):
    p=ROOT/f'experiments/results/market/{tag}.json';d=json.loads(p.read_text());assert d['status']=='completed'
    for key in ('source_sha256','input_sha256'):
        for name,digest in d.get(key,{}).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return d,str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    inputs={};rows=[]
    for seed,tag in [(19124,'curie_b10_action_geometry_20261013_v1_full'),(19129,'curie_b10_action_geometry_20261013_s19129_v1'),(19130,'curie_b10_action_geometry_20261013_s19130_v1')]:
        d,path,digest=load(tag);inputs[path]=digest
        rows.append(dict(seed=seed,nll=d['dev']['nll'],parent_nll=d['reference']['nll'],gain=d['dev_gain']))
    extra=[]
    for tag in ['curie_b10_optimizer_transport_20261013_v1_contract','curie_b10_action_tangent_20261013_v1_contract','curie_b10_sparse_action_metric_20261013_v1_contract','curie_b10_action_geometry_20261013_v1_fit']:
        d,path,digest=load(tag);inputs[path]=digest;extra.append(d)
    values=[r['gain'] for r in rows];nll=[r['nll'] for r in rows]
    packet=dict(status='completed',battle='B10',rows=rows,mean_nll=statistics.mean(nll),sd_nll=statistics.stdev(nll),mean_parent_gain=statistics.mean(values),sd_parent_gain=statistics.stdev(values),all_improve=all(g>0 for g in values),input_sha256=inputs,scope='Three independent acquired critic initializations; one common actor and reusedTRAIN/DEV8+8protocol. Each critic receives128additional causal actualfuture teacher frames then frozen16384DEV,36,820parameters versus31,804parents. Extra teacher/backward/optimizer geometry work charged separately; no TEST/fullwork/energy claim. First seed uses exact four JVP metric, confirmations contracted sparse metric with native parity contracts. Do not pool with different32targetleadingmarketprotocol.')
    path=ROOT/'experiments/results/market/curie_b10_action_geometry_20261013_v1_cohort.json'
    with path.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    # Preserve portable fitted parameters alongside ignored local checkpoints.
    import torch
    for tag in ['curie_b10_action_geometry_20261013_v1_fit','curie_b10_action_geometry_20261013_s19129_v1','curie_b10_action_geometry_20261013_s19130_v1']:
        source=ROOT/f'experiments/results/market/{tag}_critic.pt';state=torch.load(source,map_location='cpu',weights_only=False)
        entries={key:dict(shape=list(value.shape),dtype=str(value.dtype),values=value.tolist()) for key,value in state['state_dict'].items()}
        for key,entry in entries.items():
            actual=torch.tensor(entry['values'],dtype=state['state_dict'][key].dtype).reshape(entry['shape']);assert torch.equal(actual,state['state_dict'][key])
        portable=dict(status='completed',checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scale=state['scale'],state_dict=entries,scope='Exact tensor JSON roundtrip; native fitted parameters, no carried DEV history.')
        with (ROOT/f'experiments/results/market/{tag}_portable.json').open('x') as f:json.dump(portable,f);f.write('\n')
    print(json.dumps(packet),flush=True)
if __name__=='__main__':main()
