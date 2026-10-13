"""Verify completed paired native consequence teaching and portable fitted state."""
import argparse,hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();inputs={};records={}
    for name in ['curie_b10_native_antithetic_20261013_v1_smoke','curie_b10_native_antithetic_20261013_v1_fit','curie_b10_native_receipt_20261013_v2_fit']:
        path=ROOT/f'experiments/results/market/{name}.json';d=json.loads(path.read_text());assert d['status']=='completed'
        for key in ['source_sha256','input_sha256']:
            for file,digest in d.get(key,{}).items():assert sha(ROOT/file)==digest,file
        inputs[str(path.relative_to(ROOT))]=sha(path);records[name]=d
    smoke=records['curie_b10_native_antithetic_20261013_v1_smoke']
    assert all(r['paired_zero_action_parity']==0 and r['first_future_parity']==0 for r in smoke['acquisition']['rows'])
    fit=records['curie_b10_native_antithetic_20261013_v1_fit'];old=records['curie_b10_native_receipt_20261013_v2_fit'];source=fit['acquisition']
    assert source['frames']==64 and source['counterfactual_worlds']==128
    assert fit['reference']['rows']==old['reference']['rows']
    checkpoint=ROOT/'experiments/results/market/curie_b10_native_antithetic_20261013_v1_fit_critic.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    entries={key:dict(shape=list(value.shape),dtype=str(value.dtype),values=value.tolist()) for key,value in state['state_dict'].items()}
    for key,entry in entries.items():assert torch.equal(torch.tensor(entry['values'],dtype=state['state_dict'][key].dtype).reshape(entry['shape']),state['state_dict'][key])
    portable=ROOT/f'experiments/results/market/{a.tag}_portable.json'
    with portable.open('x') as f:json.dump(dict(status='completed',checkpoint_sha256=sha(checkpoint),scale=state['scale'],state_dict=entries,scope='Fitted native critic tensors; exact dtype/shape/value roundtrip, no DEV carried memory.'),f);f.write('\n')
    inputs[str(portable.relative_to(ROOT))]=sha(portable)
    result=dict(status='completed',battle='B10',nll=fit['dev']['nll'],parent_nll=fit['reference']['nll'],gain=fit['dev_gain'],previous_nll=old['dev']['nll'],teacher_improvement=old['dev']['nll']-fit['dev']['nll'],zero_action_error=0,first_update_error=0,source= {key:source[key] for key in ['frames','targets','counterfactual_worlds','counterfactual_scored_targets','counterfactual_refresh_targets','counterfactual_own_gradient_targets']},input_sha256=inputs,source_sha256={'experiments/credit/analyze_native_antithetic.py':sha(Path(__file__))},decision='Retain parent unless full prequential trajectory confirms improvement; no sealedTEST or hardware result.',scope='Same restored actor, parent,64TRAIN receipts, native residual architecture, exact DEV reference rows. Signed teacher changes source trajectory; paired worlds within each receipt. One reused DEV/initialization, extra teacher work charged.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['nll'],result['parent_nll'],result['gain'],flush=True)
if __name__=='__main__':main()
