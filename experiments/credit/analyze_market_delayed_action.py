"""Source-bound delayed native credit acquisition and learner ablation."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def load(tag):
    p=ROOT/f'experiments/results/market/{tag}.json';d=json.loads(p.read_text());assert d['status']=='completed'
    for key in ('source_sha256','input_sha256'):
        for name,digest in d.get(key,{}).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return d,str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    tags=['curie_b10_delayed_action_20261013_v1_smoke','curie_b10_delayed_action_20261013_v1_fit','curie_b10_delayed_action_20261013_v2_weighted','curie_b10_delayed_action_20261013_v3_large','curie_b10_delayed_action_20261013_v3_full','curie_b10_delayed_action_20261013_v1_ablation']
    items={};inputs={}
    for tag in tags:
        d,path,digest=load(tag);items[tag]=d;inputs[path]=digest
    short=items[tags[2]];large=items[tags[3]];full=items[tags[4]];ablation=items[tags[5]]
    assert short['acquisition']['rows']==large['acquisition']['rows'][:64]
    assert full['prefix_parity']==0 and full['dev']['targets']==16384
    curve=[dict(tag=tag,train_utility_events=items[tag]['acquisition']['targets'],nll=items[tag]['dev']['nll'],parent_nll=items[tag]['reference']['nll'],gain=items[tag]['dev_gain'],accepted=items[tag]['acquisition']['accepted'],preference_reversals=items[tag]['acquisition']['sign_reversals']) for tag in tags[1:4]]
    packet=dict(status='completed',battle='B10',curve=curve,full_nll=full['dev']['nll'],full_parent_nll=full['reference']['nll'],full_gain=full['dev_gain'],source_prefix_parity=True,dev_prefix_parity=full['prefix_parity'],ablation=dict(probes=len(ablation['rows']),mean_absolute_moment_effect=statistics.mean(abs(r['moment_effect']) for r in ablation['rows']),max_absolute_moment_effect=max(abs(r['moment_effect']) for r in ablation['rows']),learning_preference_reversals=ablation['learning_preference_reversals'],moment_preference_reversals=ablation['moment_preference_reversals'],early_moment_swap_parity=max(r['early_moment_swap_parity'] for r in ablation['rows'])),input_sha256=inputs,decision='Retain established parent critic; delayed teacher is implemented but neither small fit nor larger full deployment improves parent utility. Gain weighting and extra data improve prefix, not full trajectory.',scope='One critic/actor, reusedTRAIN/DEV. Teacherlearns finite actuallearnerconsequences, source256frames8192events; extra256alternativeworlds6144scoredtargets+4096refresh,2048own-gradienttargets subsetofscoredtargets. Existingactor/parentacquisition/native/optimizerfittingwork additional. No TEST/fullwork/scaling/publicbenchmarkwin.')
    with (ROOT/'experiments/results/market/curie_b10_delayed_action_20261013_v1_analysis.json').open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    import torch
    tag=tags[3];path=ROOT/f'experiments/results/market/{tag}_critic.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    entries={key:dict(shape=list(value.shape),values=value.tolist()) for key,value in saved['state_dict'].items()}
    for key,entry in entries.items():assert torch.equal(torch.tensor(entry['values'],dtype=saved['state_dict'][key].dtype).reshape(entry['shape']),saved['state_dict'][key])
    with (ROOT/f'experiments/results/market/{tag}_portable.json').open('x') as f:json.dump(dict(status='completed',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=saved['scale'],state_dict=entries,scope='Exact float64 fitted tensor JSON roundtrip; no carried DEV memory.'),f);f.write('\n')
    print(json.dumps(packet),flush=True)
if __name__=='__main__':main()
