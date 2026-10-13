"""Native event-race admission trained by delayed actual-Adam forecast utility."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_streaming_clock_updates import restored,load,warm,window,control
from experiments.market.paired_clock_credit import pair_window
from native_market_credit import NativeMarketCredit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def flat(state):return torch.cat([z.flatten() for z in control.tensors(state)]).detach()
def step(model,opt,grads):
    opt.zero_grad(set_to_none=True)
    for p,g in zip(model.parameters(),grads):p.grad=g.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
def proposal(model,opt,state,grads):
    actor,optimizer,numerical=copy.deepcopy((model,opt,state));step(actor,optimizer,grads);return actor,numerical

def future_score(candidate,t,m,start,group):
    model,state=candidate;values=[]
    with torch.no_grad():
        for k in range(start,start+group):
            forecast,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
            values.append(float(-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean()))
    return float(np.mean(values))

def receipt(model,old,end,own,projected,t,m,k,group,scores):
    first,last=flat(old),flat(end);descriptor=last-first;names=[n for n,_ in model.named_parameters()];info=[]
    for vectors in [own,[b-a for a,b in zip(own,projected)]]:
        by=dict(zip(names,vectors))
        for j,l in enumerate(model.layers):
            for key in ['log_rate','freq']:info.append(by[f'layers.{j}.{key}'].sum()/np.sqrt(l.n))
    descriptor=descriptor.clone();descriptor[:len(info)]=torch.stack(info).asinh()
    features=torch.cat((first,last,descriptor))[None];feedback=features.new_zeros(8)
    feedback[0]=torch.log1p((t[k+group]-t[k])/model.scale);feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+group+1],6).double().mean(0);feedback[7]=float(np.mean(scores))
    return features,torch.cat((features[0],feedback))[None,None]

FROZEN_PREFIX_RNG=None

def stream(bundle,t,m,start,blocks,group,arm,acquisition=False,teacher_interval=16):
    global FROZEN_PREFIX_RNG
    begin=time.monotonic();model,opt,state,critic,cop,scale=copy.deepcopy(bundle);carried=None;pending=None;rows=[];gates=[];labels=[];phase={k:0. for k in ['actor_credit','native_admission','proposal','future_teacher','optimizer']}
    for block in range(blocks):
        k=start+block*group;tick=time.monotonic();old=control.detach_state(state);end,own,projected,scores=pair_window(model,state,t,m,k,group);rows.append(float(np.mean(scores)));phase['actor_credit']+=time.monotonic()-tick
        # Outcomes for this window are now observed; only now may the previous decision be taught.
        if pending is not None:
            tick=time.monotonic();output,candidates=pending;delta=future_score(candidates[0],t,m,k,group)-future_score(candidates[1],t,m,k,group)
            target=float(np.clip(delta/scale,-5.,5.));cop.zero_grad(set_to_none=True);loss=(output-target).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step()
            scale=max(1e-5,.95*scale+.05*abs(delta));labels.append(dict(observed_block=block,decision_block=block-1,own_minus_projected_future_nll=delta,target=target));phase['future_teacher']+=time.monotonic()-tick;pending=None
        prediction=None
        if arm in ['online','frozen'] or acquisition:
            tick=time.monotonic();features,events=receipt(model,old,end,own,projected,t,m,k,group,scores)
            teach=(acquisition or arm=='online') and block+1<blocks and block%teacher_interval==0
            with torch.set_grad_enabled(teach):prediction,carried=critic(features,events,carried)
            gate=float(prediction.detach().item()>0.);phase['native_admission']+=time.monotonic()-tick
            if teach:
                tick=time.monotonic();candidates=[proposal(model,opt,end,g) for g in [own,projected]];pending=(prediction,candidates);phase['proposal']+=time.monotonic()-tick
        else:gate=float(arm=='projected')
        if acquisition:
            tick=time.monotonic();exact_end,grads,exact_scores=window(model,old,t,m,k,group,'bptt')
            assert max(abs(x-y) for x,y in zip(scores,exact_scores))<1e-12;phase['actor_credit']+=time.monotonic()-tick
        else:grads=[a+gate*(b-a) for a,b in zip(own,projected)]
        tick=time.monotonic();step(model,opt,grads);phase['optimizer']+=time.monotonic()-tick;state=control.detach_state(end);gates.append(gate)
        if arm=='frozen' and block==127:FROZEN_PREFIX_RNG=torch.get_rng_state().clone()
    return (model,opt,state,critic,cop,scale),dict(arm=arm,nll=float(np.mean(rows)),targets=blocks*group,wall_s=time.monotonic()-begin,phase_wall_s=phase,block_nll=rows,admission=gates,teacher_labels=labels,teachers=len(labels),native_selected_updates=critic.selected_updates,native_candidate_scores=critic.candidate_scores,native_counterfactual_values=critic.counterfactual_values)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    out=ROOT/f'experiments/results/market/{a.tag}.json';assert not out.exists();model,inputs=restored();group=8 if a.mode=='smoke' else 32;acq=4 if a.mode=='smoke' else 64;reuse=4 if a.mode=='smoke' else 512;context=128
    torch.manual_seed(19101);critic=NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,1).double();torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
    t,m,path=load('2026-08-24',context+acq*group);inputs[str(path.relative_to(ROOT))]=sha(path);opt=torch.optim.Adam(model.parameters(),lr=.0003);cop=torch.optim.Adam(critic.parameters(),lr=.001)
    # One paired sweep must recover both frozen endpoint implementations before any controller fitting.
    state=warm(model,t,m,context);end,own,projected,scores=pair_window(model,copy.deepcopy(state),t,m,context-1,group);errors={}
    for arm,pair in [('own',own),('projected',projected)]:
        _,reference,_=window(model,copy.deepcopy(state),t,m,context-1,group,arm);errors[arm]=max(float((x-y).abs().max()) for x,y in zip(pair,reference));assert errors[arm]<1e-10
    bundle,source=stream((model,opt,state,critic,cop,.001),t,m,context-1,acq,group,'online',acquisition=True,teacher_interval=1)
    t,m,path=load('2026-08-25',context+reuse*group);inputs[str(path.relative_to(ROOT))]=sha(path);bundle=(*bundle[:2],warm(bundle[0],t,m,context),*bundle[3:]);rows=[]
    for arm in ['own','projected','frozen','online']:
        if arm=='online':
            assert FROZEN_PREFIX_RNG is not None;torch.set_rng_state(FROZEN_PREFIX_RNG)
        _,row=stream(bundle,t,m,context-1,reuse,group,arm,teacher_interval=1 if a.mode=='smoke' else 16);rows.append(row)
    assert source['teachers']==acq-1 and rows[-1]['teachers']>0 and rows[-2]['teachers']==0
    gains={row['arm']:row['nll']-rows[-1]['nll'] for row in rows[:-1]}
    sources=['experiments/credit/market_native_remote_admission_longer_v2.py','experiments/credit/market_native_remote_admission_longer.py','experiments/credit/market_native_remote_admission.py','experiments/market/paired_clock_credit.py','experiments/credit/market_streaming_clock_updates.py','experiments/market/projected_clock_state.py','experiments/market/local_eligibility_state.py','experiments/credit/local_clock_eligibility.py','experiments/credit/native_market_credit.py','sleeping_machines/addressed_event_heads.py','sleeping_machines/route_activation_memory.py','sleeping_machines/parallel_head_race_language.py','sleeping_machines/sparse_race_language.py','experiments/market/NATIVE_REMOTE_ADMISSION_PROTOCOL.md','experiments/market/NATIVE_REMOTE_ADMISSION_LONGER_PROTOCOL.md','experiments/market/NATIVE_REMOTE_ADMISSION_RNG_REPAIR.md']
    prefix={}
    if a.mode=='fit':
        parent=ROOT/'experiments/results/market/curie_b10_native_admission_20261013_v1_fit.json';prior=json.loads(parent.read_text());inputs[str(parent.relative_to(ROOT))]=sha(parent)
        for row in rows:
            previous=next(x for x in prior['rows'] if x['arm']==row['arm']);error=max(abs(x-y) for x,y in zip(previous['block_nll'],row['block_nll'][:128]));assert error<1e-12;assert previous['admission']==row['admission'][:128];prefix[row['arm']]=error
    result=dict(prefix4096_block_score_max_error=prefix,status='completed' ,battle='B10',args=vars(a),acquisition=source,rows=rows,online_gain=gains,quality_gate=all(x>0 for x in gains.values()),paired_gradient_max_error=errors,source_sha256={p:sha(ROOT/p) for p in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,critic_parameters=sum(p.numel() for p in critic.parameters()),scope='Native hard-race scalar admission between own-leg and four-direction remote correction. Actor has no boundary BPTT during deployment; event-local own/projected gradients computed together. Actual endpoint Adam proposals taught by next observed32target recording-cell losses, not BPTT imitation; native banks reset at day boundary, learned weights/moments transfer. Common exact2048TRAIN actor acquisition plus native utility acquisition;16384Aug25DEV, one reused stream/actor. Always compute tangents before admission: information test, no admission compute saving. Proposal/teacher/native/optimizer costs included; original5750.687s fitting shared. No TEST, full-month, learned-basis, trading-profit or hardware claim; future gradient components approximate.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(tag=a.tag,online_gain=gains,paired_error=errors,rows=[{k:v for k,v in row.items() if k not in ['block_nll','admission','teacher_labels']} for row in rows])),flush=True)
if __name__=='__main__':main()
