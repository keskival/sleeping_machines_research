"""B10 actual Adam continuation with event-local traces and projected shared-path credit."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_native_test_time import restored,load,warm,frozen_stream
from real_boundary_stable import control
from experiments.market.local_eligibility_state import MarketLegEligibility
from experiments.market.projected_clock_state import ProjectedClockState

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def window(model,state,t,m,start,group,arm):
    params=tuple(model.parameters());names=[n for n,_ in model.named_parameters()];scores=[]
    if arm=='bptt':
        losses=[]
        for k in range(start,start+group):
            forecast,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
            loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean();scores.append(float(loss.detach()));losses.append(loss)
        grads=torch.autograd.grad(torch.stack(losses).mean(),params,allow_unused=True)
        return control.detach_state(state),[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)],scores
    projected=ProjectedClockState(model,state) if arm=='projected' else None
    def consumer(actor,time,mark,old,parameter_version=0):return projected.advance(time,mark,old,control.market_consume)
    observer=MarketLegEligibility(model,state,consumer if projected else control.market_consume) if arm in ['own','projected'] else None
    total=[torch.zeros_like(p) for p in params]
    for k in range(start,start+group):
        if projected:
            state['layers']=[z.detach().requires_grad_() for z in state['layers']];state['slots']=state['slots'].detach().requires_grad_()
        old_tensors=tuple(control.tensors(state)) if projected else ()
        forecast,end=observer.consume(t[k:k+1],m[k:k+1],state) if observer else control.market_consume(model,t[k:k+1],m[k:k+1],state)
        loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean();scores.append(float(loss.detach()))
        targets=tuple(control.tensors(end)) if observer else ()
        grads=torch.autograd.grad(loss/group,params+targets+old_tensors,allow_unused=True)
        remote=observer.historical_gradient(grads[len(params):len(params)+len(targets)]) if observer else {}
        if projected:
            correction=projected.historical_projection(grads[len(params)+len(targets):])-projected.project_parameters({n:torch.as_tensor(g) for n,g in remote.items()})
            for j,l in enumerate(model.layers):
                for offset,key in enumerate(['log_rate','freq']):remote[f'layers.{j}.{key}']+=correction[2*j+offset]/np.sqrt(l.n)
        for i,(name,p,g) in enumerate(zip(names,params,grads[:len(params)])):
            if g is not None:total[i]+=g.detach()
            if name in remote:total[i]+=torch.as_tensor(remote[name],dtype=p.dtype)
        state=control.detach_state(end)
    return state,total,scores

def stream(bundle,t,m,start,blocks,group,arm):
    model,opt,state=copy.deepcopy(bundle);scores=[];block_nll=[];step_norms=[];begin=time.monotonic();phase={'window_credit':0.,'optimizer':0.}
    for block in range(blocks):
        tick=time.monotonic();state,grads,new_scores=window(model,state,t,m,start+block*group,group,arm);phase['window_credit']+=time.monotonic()-tick
        assert all(torch.isfinite(g).all() for g in grads);scores.extend(new_scores);block_nll.append(float(np.mean(new_scores)))
        tick=time.monotonic();old=[p.detach().clone() for p in model.parameters()];opt.zero_grad(set_to_none=True)
        for p,g in zip(model.parameters(),grads):p.grad=g
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        step_norms.append(float(torch.sqrt(sum((p.detach()-x).square().sum() for p,x in zip(model.parameters(),old)))))
        phase['optimizer']+=time.monotonic()-tick
        # Numerical state persists. A new observer/zero sensitivity starts after each actual update.
    return (model,opt,state),dict(arm=arm,nll=float(np.mean(scores)),targets=len(scores),updates=blocks,wall_s=time.monotonic()-begin,phase_wall_s=phase,block_nll=block_nll,step_norms=step_norms)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic();out=ROOT/f'experiments/results/market/{a.tag}.json';assert not out.exists()
    model,inputs=restored();context=128;group=1 if a.mode=='smoke' else 32;acq=2 if a.mode=='smoke' else 64;reuse=3 if a.mode=='smoke' else 128
    t,m,path=load('2026-08-24',context+acq*group);inputs[str(path.relative_to(ROOT))]=sha(path)
    opt=torch.optim.Adam(model.parameters(),lr=.0003);bundle,acquisition=stream((model,opt,warm(model,t,m,context)),t,m,context-1,acq,group,'bptt')
    t,m,path=load('2026-08-25',context+reuse*group);inputs[str(path.relative_to(ROOT))]=sha(path);bundle=(bundle[0],bundle[1],warm(bundle[0],t,m,context))
    rows=[frozen_stream(copy.deepcopy(bundle[0]),t,m,copy.deepcopy(bundle[2]),context-1,reuse,group)];finals={}
    for arm in ['bptt','local','own','projected']:
        final,row=stream(bundle,t,m,context-1,reuse,group,arm);rows.append(row);finals[arm]=final
    parity={}
    if a.mode=='smoke':
        reference=finals['bptt']
        for arm,final in finals.items():
            parameter=max(float((p-q).abs().max()) for p,q in zip(reference[0].parameters(),final[0].parameters()))
            moments=max(float((reference[1].state[p][key]-final[1].state[q][key]).abs().max()) for p,q in zip(reference[0].parameters(),final[0].parameters()) for key in ['exp_avg','exp_avg_sq','step'])
            score=max(abs(x-y) for x,y in zip(rows[1]['block_nll'],next(r for r in rows if r['arm']==arm)['block_nll']))
            assert max(parameter,moments,score)<1e-10;parity[arm]=dict(parameter_max_error=parameter,adam_state_max_error=moments,score_max_error=score)
    by={r['arm']:r for r in rows};gains={key:by[key]['nll']-by['projected']['nll'] for key in by if key!='projected'}
    sources=['experiments/credit/market_streaming_clock_updates.py','experiments/market/projected_clock_state.py','experiments/market/local_eligibility_state.py','experiments/credit/local_clock_eligibility.py','experiments/credit/market_native_test_time.py','experiments/credit/real_boundary_stable.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py','experiments/market/STREAMING_CLOCK_UPDATES_PROTOCOL.md']
    r=dict(status='completed',battle='B10',args=vars(a),rows=rows,acquisition=acquisition,projected_gain=gains,one_event_adam_parity=parity,quality_gate=all(v>0 for v in gains.values()),source_sha256={n:sha(ROOT/n) for n in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One restored market actor; common exact2048TRAIN acquisition in fit,4096Aug25DEV score-before-update forecasts; 32event fixed-parameter windows with numerical state persistence and traces reset after actual Adam updates. Fully frozen, exact BPTT, event-local, own traces and four shared-path clock projections. No native critic/residual teacher yet. NumPy correctness backend, includes trace/optimizer work; original5750.687s fitting additionally charged to every arm. Reused development stream, no sealed TEST or trading-profit claim. Smoke uses one-event windows to check actual parameters/Adam moments/scores across updates; longer-window projected gradients remain approximate.')
    with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(dict(tag=a.tag,projected_gain=gains,rows=[{k:v for k,v in row.items() if k not in ['block_nll','step_norms']} for row in rows])),flush=True)
if __name__=='__main__':main()
