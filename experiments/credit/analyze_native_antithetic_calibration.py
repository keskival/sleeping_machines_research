"""Source-bound critic clipping comparison and portable state."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();items=[];inputs={}
    tags=['curie_b10_native_antithetic_20261013_v1_fit','curie_b10_native_antithetic_20261013_v2_unclipped_fit','curie_b10_native_antithetic_20261013_v2_unclipped_smoke']
    for tag in tags:
        path=ROOT/f'experiments/results/market/{tag}.json';d=json.loads(path.read_text());assert d['status']=='completed'
        for key in ('source_sha256','input_sha256'):
            for file,digest in d.get(key,{}).items():assert sha(ROOT/file)==digest,file
        inputs[str(path.relative_to(ROOT))]=sha(path);items.append(d)
    clipped,unclipped,smoke=items
    assert clipped['reference']['rows']==unclipped['reference']['rows']
    assert all(r['paired_zero_action_parity']==0 and r['first_future_parity']<1e-9 for r in smoke['acquisition']['rows'])
    # Identical issued action/teacher before the first changed optimizer step.
    assert clipped['acquisition']['rows'][0]['antithetic_contrast']==unclipped['acquisition']['rows'][0]['antithetic_contrast']
    rows=[]
    for label,d in [('clip1',clipped),('unclipped',unclipped)]:
        source=d['acquisition']['rows'];rows.append(dict(critic_update=label,nll=d['dev']['nll'],parent_gain=d['dev_gain'],mean_gradient_norm=statistics.mean(r['critic_grad_norm'] for r in source),above_norm1=sum(r['critic_grad_norm']>1 for r in source),quarter_gains=[statistics.mean(.5*(a['producer_nll']+a['future_nll']-b['producer_nll']-b['future_nll']) for a,b in zip(d['reference']['rows'][k:k+32],d['dev']['rows'][k:k+32])) for k in range(0,128,32)]))
    path=ROOT/f'experiments/results/market/{tags[1]}_critic.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    entries={key:dict(shape=list(v.shape),dtype=str(v.dtype),values=v.tolist()) for key,v in saved['state_dict'].items()}
    for key,e in entries.items():assert torch.equal(torch.tensor(e['values'],dtype=saved['state_dict'][key].dtype).reshape(e['shape']),saved['state_dict'][key])
    portable=ROOT/f'experiments/results/market/{a.tag}_portable.json'
    with portable.open('x') as f:json.dump(dict(status='completed',checkpoint_sha256=sha(path),scale=saved['scale'],state_dict=entries,scope='Exact tensor roundtrip; no carried DEV memory.'),f);f.write('\n')
    inputs[str(portable.relative_to(ROOT))]=sha(portable)
    result=dict(status='completed',battle='B10',rows=rows,unclipped_gain_over_clipped=clipped['dev']['nll']-unclipped['dev']['nll'],initial_contrast_parity=True,input_sha256=inputs,source_sha256={'experiments/credit/analyze_native_antithetic_calibration.py':sha(Path(__file__))},scope='One64receipt source/2048TRAIN and2048reusedDEV perarm; only critic gradient clipping changes. Actor clip1, native race/state/receipt architecture, action bound, seeds/data/reference fixed. No sealedTEST/fulltrajectory/hardwareclaim. Additional128worlds perarm and parent/actor acquisition/native work charged.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
