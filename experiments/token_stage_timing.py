"""Low-overhead CPU phase timing of an unchanged completed fit; guarded execution only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import torch
import horizon_token_language_engine as engine
import horizon_token_language as selection
from sleeping_machines.token_readout import TokenReadout

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--tag',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    control_path=Path(a.control);control=json.loads(control_path.read_text());assert control['status']=='completed'
    for name,digest in control['identity']['source_sha256'].items():assert engine.sha(ROOT/name)==digest,name
    args=[]
    for key,value in control['args'].items():
        if key in ('tag','resume','stop_after_step') or value is None:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:args.append(flag)
        else:args.extend([flag,str(value)])
    sys.argv=['horizon_token_language.py','--tag',a.tag,*args]
    times={};calls={};active=False;mode='factual';started=None;whole=0.;updates=0
    def add(name,duration):
        times[name]=times.get(name,0.)+duration;calls[name]=calls.get(name,0)+1
    old_forward=engine.Model.forward;old_nll=TokenReadout.nll;old_backward=torch.Tensor.backward
    old_zero=torch.optim.AdamW.zero_grad;old_step=torch.optim.AdamW.step
    def forward(self,*args,**kwargs):
        nonlocal mode
        mode='counterfactual' if self.shadow_mode else 'factual';name=mode+'_core_forward'
        before=time.perf_counter()
        result=old_forward(self,*args,**kwargs)
        if active:add(name,time.perf_counter()-before)
        return result
    def nll(self,*args,**kwargs):
        before=time.perf_counter();result=old_nll(self,*args,**kwargs)
        if active:add(mode+'_readout',time.perf_counter()-before)
        return result
    def backward(self,*args,**kwargs):
        before=time.perf_counter();result=old_backward(self,*args,**kwargs)
        if active:add('backward',time.perf_counter()-before)
        return result
    def zero(self,*args,**kwargs):
        nonlocal active,started
        active=True;started=time.perf_counter();return old_zero(self,*args,**kwargs)
    def step(self,*args,**kwargs):
        nonlocal active,whole,updates
        before=time.perf_counter();result=old_step(self,*args,**kwargs);after=time.perf_counter()
        add('optimizer_step',after-before);whole+=after-started;updates+=1;active=False;return result
    engine.Model.forward=forward;TokenReadout.nll=nll;torch.Tensor.backward=backward
    torch.optim.AdamW.zero_grad=zero;torch.optim.AdamW.step=step
    try:selection.main()
    finally:
        engine.Model.forward=old_forward;TokenReadout.nll=old_nll;torch.Tensor.backward=old_backward
        torch.optim.AdamW.zero_grad=old_zero;torch.optim.AdamW.step=old_step
    result_path=ROOT/f'experiments/results/token_language/{a.tag}.json';result=json.loads(result_path.read_text())
    assert result['status']=='completed' and updates==control['args']['steps']
    assert result['presentations_total']==control['presentations_total']
    assert len(result['curve'])==len(control['curve'])
    errors=[abs(x['dev_nll']-y['dev_nll']) for x,y in zip(result['curve'],control['curve'])]
    assert max(errors)<2e-6
    accounted=sum(times.values());assert accounted<=whole+1e-6
    output.write_text(json.dumps(dict(status='completed',control=str(control_path),control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),audited_tag=a.tag,
        optimizer_updates=updates,fitting_targets=result['presentations_total'],phase_wall_s=times,phase_calls=calls,
        whole_fitting_wall_s=whole,other_in_step_wall_s=whole-accounted,curve_parity_max_error=max(errors),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One-thread CPU perf_counter intervals, zero_grad through AdamWstep. Core includes token interface, temporal operations, keys/writes and message/state processing. Readout is exact adaptive NLL. Backward covers all gradients; optimizer separate. Other covers teacher sampling/weighting, reductions, clipping and in-step diagnostics. Preprocessing/evaluation/post-step diagnostics/serialization excluded. Wrappers add timing overhead; arithmetic trace wall is not used. No hardware energy, FLOP reduction or quality gain inferred.'),indent=2)+'\n')
if __name__=='__main__':main()
