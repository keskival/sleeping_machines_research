"""B10 primary chronological online comparison; forecast, reveal, then learn."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(Path(__file__).parent))
from capability_admission import validate
from b10_matched_data import prepare,digest
from b10_matched_models import MatchedRace
from b10_online_stream import forecast,revealed_terms,detach
from b10_operator_ledger_v3 import Ledger

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def apply(model,opt,t,m,nt,nm,state,context):
    opt.zero_grad(set_to_none=True);parameters,state=forecast(model,t,m,state,context);terms=revealed_terms(model,parameters,t,nt,nm);loss=-terms.mean();assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);opt.step();return terms.detach(),detach(state)

def sources():
    names=['experiments/market/b10_online_comparison.py','experiments/market/b10_online_stream.py','experiments/capability_admission.py','experiments/market/b10_matched_models.py','experiments/market/b10_stable_clock_head.py','experiments/market/b10_operator_ledger_v3.py','experiments/market/b10_depth_tpp_v1.py','experiments/tpp/race_tpp_v5.py']
    return {n:digest(ROOT/n) for n in names}

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['gate','online'],required=True);p.add_argument('--admission',required=True);p.add_argument('--family',choices=['native','lstm','transformer']);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);began=time.monotonic();cpu=time.process_time();contract=json.loads((ROOT/a.admission).read_text());validate(contract)
    for n,d in contract['source_sha256'].items():assert digest(ROOT/n)==d,n
    data,receipt=prepare();gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0];rows=[];inputs={a.admission:digest(ROOT/a.admission)};context=contract['transformer_context'];chunk=contract['update_targets'];targets=contract['scored_targets']
    families=('native','lstm','transformer') if a.mode=='gate' or a.family is None else (a.family,)
    for family in families:
        torch.manual_seed(0);model=MatchedRace(gaps,family);model.eval() # Dropout off, gradients/Adam ON for every family.
        if a.mode=='gate':
            t=torch.tensor(data['train'][0][0][:65])[None];m=torch.tensor(data['train'][0][1][:65])[None]
            with torch.no_grad():
                parameters,state=forecast(model,t[:,:64],m[:,:64],None,context);terms=revealed_terms(model,parameters,t[:,:64],t[:,1:65],m[:,1:65]);reference=model.terms(t,m);assert torch.allclose(terms,reference,rtol=1e-10,atol=1e-10),family+' original law'
                p1,s1=forecast(model,t[:,:32],m[:,:32],None,context);p2,s2=forecast(model,t[:,32:64],m[:,32:64],s1,context);assert all(torch.allclose(x[:,32:],y,rtol=1e-10,atol=1e-10) for x,y in zip(parameters,p2)),family+' numerical state carry'
                # Future input mutation cannot change any earlier issued parameter.
                altered_t=t[:,:64].clone();altered_m=m[:,:64].clone();altered_t[:,33:]+=100.;altered_m[:,33:]=(altered_m[:,33:]+1)%6;other,_=forecast(model,altered_t,altered_m,None,context);assert all(torch.equal(x[:,:33],y[:,:33]) for x,y in zip(parameters,other)),family+' causality'
            opt=torch.optim.AdamW(model.parameters(),lr=contract['online_lr'],weight_decay=1e-4,foreach=False);initial_model=copy.deepcopy(model.state_dict());initial_opt=copy.deepcopy(opt.state_dict());ledger=Ledger()
            with ledger:loss,st=apply(model,opt,t[:,:32],m[:,:32],t[:,1:33],m[:,1:33],None,context)
            expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict());assert any(not torch.equal(initial_model[n],v) for n,v in expected.items()),family+' weights did not adapt';model.load_state_dict(initial_model);opt.load_state_dict(initial_opt);again,st2=apply(model,opt,t[:,:32],m[:,:32],t[:,1:33],m[:,1:33],None,context);assert equal(loss,again) and equal(st,st2) and equal(expected,model.state_dict()) and equal(expected_opt,opt.state_dict())
            if family=='transformer':
                # Measure actual larger context update before selecting it for fits.
                base=data['train'][0];n=context+chunk+1;tt=torch.arange(n,dtype=torch.float64)[None]*.001;mm=torch.arange(n)[None]%6
                with torch.no_grad():_,history=forecast(model,tt[:,:context],mm[:,:context],None,context)
                probe=Ledger()
                with probe:apply(model,opt,tt[:,context:-1],mm[:,context:-1],tt[:,context+1:],mm[:,context+1:],detach(history),context)
            rows.append(dict(family=family,legacy_likelihood_parity=True,persistent_state_partition_parity=True,future_input_invariance=True,actual_online_weight_change=True,actual_update_recovery=True,measured_update_ledger=ledger.record()))
            continue
        gate_path=ROOT/contract['gate_result'];gate=json.loads(gate_path.read_text());assert gate['status']=='completed' and gate['mode']=='gate';assert gate['source_sha256']==sources();inputs[contract['gate_result']]=digest(gate_path)
        base=ROOT/f'experiments/results/market/aws_b10_stable_{family}_s0_20261014_v1.json';fit=json.loads(base.read_text());assert fit['status']=='completed';weights=ROOT/fit['final_weights'];saved=torch.load(weights,weights_only=False);model.load_state_dict(saved['model']);inputs[str(base.relative_to(ROOT))]=digest(base);inputs[str(weights.relative_to(ROOT))]=digest(weights)
        # Selected best weights lack co-saved moments. Initialize all online
        # learners identically, then pay the same chronological TRAIN warmup.
        opt=torch.optim.AdamW(model.parameters(),lr=contract['online_lr'],weight_decay=1e-4,foreach=False);before={n:v.detach().clone() for n,v in model.named_parameters()};state=None;warm_ledger=Ledger();online_ledger=Ledger();curve=[]
        z=np.load(ROOT/'data/binance/npz/2026-08-24.npz');origin=int(z['t'][0]);t=torch.tensor((z['t'][:2049]-origin)*1e-6)[None];m=torch.tensor(z['mark'][:2049].astype(np.int64))[None]
        with warm_ledger:
            for cursor in range(0,2048,chunk):_,state=apply(model,opt,t[:,cursor:cursor+chunk],m[:,cursor:cursor+chunk],t[:,cursor+1:cursor+chunk+1],m[:,cursor+1:cursor+chunk+1],state,context)
        z=np.load(ROOT/'data/binance/npz/2026-08-25.npz');t=torch.tensor((z['t'][:targets+128]-origin)*1e-6)[None];m=torch.tensor(z['mark'][:targets+128].astype(np.int64))[None]
        with torch.no_grad():_,state=forecast(model,t[:,:127],m[:,:127],state,context);state=detach(state)
        total=0.;count=0;started=time.monotonic()
        with online_ledger:
            for cursor in range(127,127+targets,chunk):
                # Input event j issues the law for j+1. Future outcomes cannot
                # affect that law; one Adam step follows each revealed32 events.
                loss,state=apply(model,opt,t[:,cursor:cursor+chunk],m[:,cursor:cursor+chunk],t[:,cursor+1:cursor+chunk+1],m[:,cursor+1:cursor+chunk+1],state,context);total-=float(loss.double().sum());count+=loss.numel()
                if count%4096==0:curve.append(dict(targets=count,nll=total/count));print(json.dumps(dict(family=family,**curve[-1])),flush=True)
        assert count==targets;assert any(not torch.equal(v,before[n]) for n,v in model.named_parameters())
        out=ROOT/f'experiments/results/market/{a.tag}_{family}.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,scored_targets=count),out)
        rows.append(dict(family=family,seed=0,nll=total/count,targets=count,curve=curve,online_weight_updates=targets//chunk,persistent_numeric_state=True,online_wall_s=time.monotonic()-started,warmup_targets=2048,warmup_ledger=warm_ledger.record(),online_ledger=online_ledger.record(),inherited_fitting_arithmetic=fit['fitting_arithmetic'],inherited_fitting_wall_s=fit['wall_s'],final_learner=str(out.relative_to(ROOT))))
    result=dict(status='completed',battle='B10',mode=a.mode,capability_admission_passed=True,rows=rows,source_sha256=sources(),input_sha256=inputs,wall_s=time.monotonic()-began,cpu_s=time.process_time()-cpu,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Primary chronological online comparison: every family adapts weights after revealed32-target blocks; native/LSTM numerical state persists, Transformer retains causal rolling history plus persistent addressed output state. TRAIN Aug24 warmup2048; DEV Aug25 context128 then16384 adjacent forecasts, no repeated window resets. Seed0, reused DEV, Septembersealed. Fresh equally initialized online Adam moments for selected fitted weights; no claim that acquired moments were restored. Full costs reported; matched observed stream/update opportunity, not equal adaptation FLOPs. Distinct from earlier page11 native learning-rule cohort; no direct cross-protocol score comparison.')
    out=ROOT/f'experiments/results/market/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
