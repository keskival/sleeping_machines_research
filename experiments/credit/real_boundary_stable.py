"""B10 integrated clock-read repair and learned-credit comparison.
Keep positive addressed state untouched; compress only its clock-head read.
"""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
import real_learned_boundary_v2 as control
import real_learned_boundary_v3 as learned
ROOT=Path(__file__).resolve().parents[2]
class LogSlotClock(torch.nn.Linear):
    def __init__(self,old,d):
        super().__init__(old.in_features,old.out_features,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype);self.d=d;self.load_state_dict(old.state_dict())
    def forward(self,x):
        return torch.nn.functional.linear(torch.cat((x[...,:self.d],torch.log1p(x[...,self.d:])),dim=-1),self.weight,self.bias)
original_factory=control.model_data
def model_data(seed,n,context):
    model,times,marks=original_factory(seed,n,context);model.clock=LogSlotClock(model.clock,model.embed.embedding_dim);return model,times,marks
control.model_data=model_data;learned.model_data=model_data

def contract():
    result=control.contract();model,times,marks=model_data(183,80,32)
    mask=torch.ones_like(marks[None],dtype=torch.bool);h,slots=model.encode(times[None],marks[None],mask);params=model.clocks(h[:,:-1],slots[:,:-1]);g=times[1:]-times[:-1];cell=control.market.CELL
    _,s0=model.clock_terms(g[None],*params[:4]);_,s1=model.clock_terms((g+cell)[None],*params[:4]);hazard,_=model.clock_terms((g+cell/2)[None],*params[:4]);selected=params[4].gather(-1,marks[None,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1)
    full=control.market.log_interval(s0.sum(-1),s1.sum(-1))+torch.logsumexp(hazard+selected,-1)-torch.logsumexp(hazard,-1)
    model.zero_grad(set_to_none=True);(-full.mean()).backward();gradient=control.gradients(model).clone();state=None;stream=[]
    for k in range(len(times)-1):
        forecast,state=control.market_consume(model,times[k:k+1],marks[k:k+1],state);stream.append(control.revealed_log_probability(model,forecast,times[k+1:k+2],marks[k+1:k+2]).mean())
    stream=torch.stack(stream);score_error=float((stream-full[0]).abs().max());model.zero_grad(set_to_none=True);(-stream.mean()).backward();gradient_error=float((gradient-control.gradients(model)).abs().max());assert score_error<1e-8 and gradient_error<1e-8,(score_error,gradient_error)
    x=torch.tensor([0.,1.,1e6],dtype=torch.float64,requires_grad=True);y=torch.log1p(x);derivative=torch.autograd.grad(y.sum(),x)[0];assert torch.allclose(derivative,1/(1+x))
    result.update(full_stream_score_error=score_error,full_stream_gradient_error=gradient_error,readout_derivative='1/(1+positive_slot); numerical state and decay/write law unchanged',variant='positive_slot_log_clock');return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','measure'],required=True);ap.add_argument('--seed',type=int,default=183);a=ap.parse_args();a.targets=8192;a.context=128;a.group=32;a.audit_probability=.25;torch.set_num_threads(1);start=time.monotonic();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists()
    if a.mode=='contract':metrics=contract()
    else:
        if a.mode=='smoke':a.targets=64;a.context=32;a.group=8
        rows=[control.run(a,arm,a.seed) for arm in ('bptt','untrained_audit')]+[learned.run(a,'calibrated_audit',a.seed)];assert all(r['targets']==a.targets for r in rows);assert rows[1]['audits']==rows[2]['audits'];metrics=dict(rows=rows)
    sources=['experiments/credit/real_boundary_stable.py','experiments/credit/real_learned_boundary_v2.py','experiments/credit/real_learned_boundary_v3.py','experiments/credit/real_online_episode_bridge.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py']
    out.write_text(json.dumps(dict(status='completed',tag=a.tag,battle='B10/R1',model_variant='positive_slot_log_clock',args=vars(a),metrics=metrics,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Changed native model, all three learner arms use identical log-compressed positive-slot clock read. Original results remain evidence, not replaced. Same real TRAIN fragment; no TEST, sparse-hard-route or optimized efficiency claim. Full local derivatives/replay/audit/optimizer work charged.'),indent=2)+'\n')
if __name__=='__main__':main()
