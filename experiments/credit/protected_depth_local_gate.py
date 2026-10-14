"""R1 warmed protected-depth TRAIN calibration and exact local-update recovery."""
import argparse,copy,json,math,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(Path(__file__).parent))
import protected_depth_learning_pilot as pilot
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.learning.calibrated_receiver_adamw import CalibratedReceiverBankAdamW
from sleeping_machines.learning.calibrated_bank_caps import calibrate_bank_caps

PILOT='aws_protected_depth_strong_parent_pilot_20261014_v1'

def restore_arm(label):
    base,r,args,train,counts,order,parent,dense,old=pilot.load_acquired()
    if label=='grown':
        config=SimpleNamespace(**vars(args));config.depth+=2
        with torch.random.fork_rng():torch.manual_seed(args.seed+17);fresh=pilot.driver.Model(config,order,counts)
        model=pilot.grow_protected_language(parent,fresh)
        for name,offset in [('new_teacher_generator',9),('new_site_generator',8),('new_alternative_generator',10),('new_position_generator',11)]:setattr(model,name,torch.Generator().manual_seed(args.seed+offset))
    else:model=parent;config=args
    path=ROOT/f'experiments/results/token_language/{PILOT}_{label}.pt';ck=torch.load(path,weights_only=False);model.load_state_dict(ck['model'])
    for name,rng in ck['owned_rng'].items():getattr(model,name).set_state(rng)
    opt=ReceiverBankAdamW(model,lr=config.lr,weight_decay=config.weight_decay);opt.load_state_dict(ck['optimizer'])
    return model,opt,config,train,ck,path

def private_state(model):
    names=('site_generator','position_generator','local_generator','alternative_generator','new_teacher_generator','new_site_generator','new_alternative_generator','new_position_generator')
    return {name:getattr(model,name).get_state().clone() for name in names if hasattr(model,name)}

def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic();rows=[];inputs={}
    receipt=ROOT/f'experiments/results/diagnostics/{PILOT}.json';record=json.loads(receipt.read_text());assert record['status']=='completed'
    for name,digest in record['source_sha256'].items():assert pilot.driver.sha(ROOT/name)==digest,name
    inputs[str(receipt.relative_to(ROOT))]=pilot.driver.sha(receipt)
    for label in ('parent','grown'):
        model,opt,args,train,ck,path=restore_arm(label);inputs[str(path.relative_to(ROOT))]=pilot.driver.sha(path);model.train();initial=copy.deepcopy(model.state_dict());state=pilot.clone_state(ck['state']);gen=torch.Generator().set_state(ck['generator']);teacher=torch.Generator().set_state(ck['teacher_generator']);tokens=pilot.driver.BlockLaneTokens(train,args.train_tokens,args.lanes,args.exposure,args.steps+96,args.chunk);tokens.block=ck['training_block'];cursor=ck['cursor'];norms={b:[] for b in opt.receiver.mapping};scales=[];held=[];work=[];internal_norms=[]
        for index in range(32):
            if cursor>=tokens.shape[1]-1:cursor=0;state=None;tokens.advance()
            n=min(args.chunk,tokens.shape[1]-1-cursor);buf=tokens[:,cursor:cursor+n+1];opt.zero_grad();loss,state,audits=pilot.raw_step(model,state,gen,teacher,args,buf,ck['step']+index+1);cursor+=n;work.extend(audits)
            raw={p.bank:p.gradient for p in opt.receiver.packets('calibration')};total=math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None));scale=min(1.,1/(total+1e-6))
            if label=='grown':internal_norms.append(math.sqrt(sum(float(p.grad[model.core.parent_depth:].double().square().sum()) for p in model.core.banks.values() if p.grad is not None)))
            if index<24:
                scales.append(scale)
                for bank in norms:norms[bank].append(float(raw[bank].double().norm()) if bank in raw else 0.)
            else:held.append({bank:float(g.double().norm()) for bank,g in raw.items()})
        # Calibration evolves numerical states and owned noise, never actor/moments.
        now=model.state_dict();assert all(equal_tree(v,now[k]) for k,v in initial.items() if k!='core._extra_state')
        fitted=calibrate_bank_caps(norms,scales,quantile=.9,budget=1.25)
        observed=sum(len(row) for row in held);clipped=sum(norm>fitted['caps'][bank] for row in held for bank,norm in row.items());assert clipped>0,'Local policy did not clip'
        if label=='grown':assert max(internal_norms)>0,'Added temporal stages received no factual credit after warmup'
        # Serialize actor, receiver moments/versions, numeric state and every RNG at
        # this issued cut; replay a REAL local update in two independent learners.
        opt.zero_grad();local=CalibratedReceiverBankAdamW(model,fitted['caps'],lr=args.lr,weight_decay=args.weight_decay);local.load_state_dict({**opt.state_dict(),'bank_caps':fitted['caps']})
        twin=copy.deepcopy(model);other=CalibratedReceiverBankAdamW(twin,fitted['caps'],lr=args.lr,weight_decay=args.weight_decay);other.load_state_dict(copy.deepcopy(local.state_dict()))
        if cursor>=tokens.shape[1]-1:cursor=0;state=None;tokens.advance()
        n=min(args.chunk,tokens.shape[1]-1-cursor);buf=tokens[:,cursor:cursor+n+1];out=[]
        for actor,learner in ((model,local),(twin,other)):
            g=torch.Generator().set_state(gen.get_state());t=torch.Generator().set_state(teacher.get_state());learner.zero_grad();loss,s,audits=pilot.raw_step(actor,pilot.clone_state(state),g,t,args,buf,ck['step']+33);learner.step();learner.zero_grad();out.append((loss,s,audits,g.get_state(),t.get_state(),private_state(actor)))
        assert equal_tree(out[0],out[1]);assert equal_tree(model.state_dict(),twin.state_dict());assert equal_tree(local.state_dict(),other.state_dict())
        rows.append(dict(arm=label,calibration=fitted,heldout_train_clip_fraction=clipped/observed,heldout_observed_banks=observed,new_internal_raw_gradient_norm_max=max(internal_norms) if internal_norms else None,actual_local_update_full_recovery=True,teacher_work=work,calibration_targets=32*args.chunk*args.lanes,recovery_update_targets=2*n*args.lanes))
    names=['experiments/credit/protected_depth_local_gate.py',*record['source_sha256'],'sleeping_machines/learning/calibrated_receiver_adamw.py','sleeping_machines/learning/calibrated_bank_caps.py']
    result=dict(status='completed',battle='R1',rows=rows,input_sha256=inputs,source_sha256={n:pilot.driver.sha(ROOT/n) for n in names},wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Warmed64-update TRAIN pilot learners;24 calibration+8 heldoutTRAIN chunks per arm, actor/moments frozen during calibration. Fixed caps cover exact actual schema, no reused depth8 caps. Actual local update recovery includes all receiver moments/versions, numerical state and RNG. Offline global norm paid; no DEVselection, quality win, fully asynchronous runtime or hardware speed claim. Full comparison starts from pilot cut and charges calibration exposure/work separately.')
    path=ROOT/f'experiments/results/credit/{a.tag}.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
