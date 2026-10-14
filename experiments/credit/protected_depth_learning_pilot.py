"""R1 integrated strong-parent growth pilot with supported old/new route credit.

Preserve old per-stage teacher opportunity: original stream audits old stages,
independent equally frequent stream audits added stages. No fixed-budget depth
credit dilution. Both streams use recorded support and paired owned clock noise.
Globalclip1 here is a numerical/resource pilot; local calibration precedes fits.
"""
import argparse,copy,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import multidepth_readout_token_language as driver
from sleeping_machines.protected_depth_language import grow_protected_language,clone_state
from sleeping_machines.learning.dense_to_receiver_moments import acquired_receiver_optimizer
from sleeping_machines.sparse_counterfactual_episodes import detach
from sleeping_machines.event_credit_sites import sample_site
from sleeping_machines.sampled_suffix_utility import sample_positions,suffix_difference
from sleeping_machines.paired_route_credit import paired_route_credit


def raw_step(model,state,gen,teacher_gen,args,buf,step,with_new=True):
    n=buf.shape[1]-1;grown=hasattr(model.core,'parent_depth');old_depth=model.core.parent_depth if grown else args.depth
    audits=[]
    if torch.rand((),generator=teacher_gen)<args.teacher_probability:
        token,d,h,inv=sample_site(n,old_depth,args.heads,model.site_generator);audits.append((token,d,h,inv,'old'))
    if grown and with_new and torch.rand((),generator=model.new_teacher_generator)<args.teacher_probability:
        token,d,h,inv=sample_site(n,args.depth-old_depth,args.heads,model.new_site_generator);audits.append((token,d+old_depth,h,inv,'new'))
    model.suppress_site=audits[0][:3] if audits else None
    values,state,pis=model(buf[:,:-1],state,gen,args);terms=model.readout.nll(values,buf[:,1:]);future=0.;rows=[]
    factual_branch_rng=model.core.new_depth_generator.get_state() if grown else None
    for token,d,h,inv,kind in audits:
        pi=pis[token*args.depth+d][1][:,h,:].double();winner=state['race_winners'][token,d,:,h];eligible=1-torch.nn.functional.one_hot(winner,args.pool).double();proposal=.9*pi.detach()*eligible/(pi.detach()*eligible).sum(-1,keepdim=True)+.1*eligible/(args.pool-1)
        alt_gen=model.alternative_generator if kind=='old' else model.new_alternative_generator
        alt=torch.multinomial(proposal,1,generator=alt_gen).squeeze(1);q=proposal.gather(1,alt[:,None]).squeeze(1)
        begin=(token//args.credit_window)*args.credit_window
        window=model.replay_windows[begin] if grown else model.replay_window
        model.force_site=(token-begin,d,h,alt);model.shadow_mode=True
        shadow_gen=torch.Generator().set_state(window['rng'])
        pos_gen=model.position_generator if kind=='old' else model.new_position_generator
        end=min(n,(token//args.credit_window+1)*args.credit_window);positions=sample_positions(end-token,args.future_score_positions,pos_gen)+token;length=int(positions[-1])+1
        if grown:model.core.new_depth_generator.set_state(window['new_depth_rng'])
        with torch.no_grad():
            shadow,_,_=model(buf[:,begin:length],window['state'],shadow_gen,args)
            difference=suffix_difference(model.readout,shadow,buf[:,begin+1:],terms.reshape(args.lanes,n)[:,begin:],positions-begin)
        if grown:model.core.new_depth_generator.set_state(factual_branch_rng)
        model.shadow_mode=False
        weight=inv*(end-token)/n;future=future+weight*paired_route_credit(pi,alt,q,difference)/args.teacher_probability
        rows.append(dict(kind=kind,site=[token,d,h],inverse_site_probability=inv,proposal_probability=q.tolist(),mean_suffix_loss_delta=float(difference.mean()),replayed_targets=(length-begin)*args.lanes,scored_targets=len(positions)*args.lanes))
    model.force_site=model.suppress_site=None;model.replay_window=None
    if grown:model.replay_windows={}
    objective=terms.mean()+args.future_scale*future
    assert torch.isfinite(objective);objective.backward()
    return terms.detach(),detach(state),rows


def load_acquired():
    base=ROOT/'experiments/results/token_language/aws_multidepth_readout_8m_20261013_s6_v1';record=json.loads(base.with_suffix('.json').read_text());assert record['status']=='completed'
    for path,digest in record['source_sha256'].items():assert driver.sha(ROOT/path)==digest,path
    args=SimpleNamespace(**record['args']);train=driver.load_tokens(args.train_file);counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True);torch.manual_seed(args.seed);parent=driver.Model(args,order,counts);saved=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(saved['model']);
    for name in ('site_generator','position_generator','local_generator','alternative_generator'):getattr(parent,name).set_state(saved[name])
    dense=torch.optim.AdamW(parent.parameters(),lr=args.lr,weight_decay=args.weight_decay);dense.load_state_dict(saved['optimizer'])
    return base,record,args,train,counts,order,parent,dense,saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--updates',type=int,default=64);a=p.parse_args();assert 1<=a.updates<=64;torch.set_num_threads(1);start=time.monotonic();base,r,args,train,counts,order,parent,dense,saved=load_acquired();grown_args=SimpleNamespace(**vars(args));grown_args.depth=args.depth+2
    with torch.random.fork_rng():torch.manual_seed(args.seed+17);fresh=driver.Model(grown_args,order,counts)
    rng=torch.get_rng_state();candidate=grow_protected_language(parent,fresh);assert torch.equal(rng,torch.get_rng_state());candidate.new_teacher_generator=torch.Generator().manual_seed(args.seed+9);candidate.new_site_generator=torch.Generator().manual_seed(args.seed+8);candidate.new_alternative_generator=torch.Generator().manual_seed(args.seed+10);candidate.new_position_generator=torch.Generator().manual_seed(args.seed+11)
    # Boundary contract on a fresh TRAIN panel, before fitting either arm.
    panel=driver.LaneTokens(train,262144,136,args.lanes)[:,:];gp=torch.Generator().manual_seed(93);gc=torch.Generator().manual_seed(93)
    parent.eval();candidate.eval();xp,sp,_=parent(panel[:,:16],None,gp,args);xc,sc,_=candidate(panel[:,:16],None,gc,grown_args)
    lp=parent.readout.nll(xp,panel[:,1:17]);lc=candidate.readout.nll(xc,panel[:,1:17]);assert torch.equal(lp,lc),'Strong parent prediction lost on insertion'
    assert torch.equal(gp.get_state(),gc.get_state()) and torch.equal(sp['ctx_arr'],sc['ctx_arr'])
    names=dict(parent.named_parameters());old_grad=torch.autograd.grad(lp.mean(),tuple(names.values()),allow_unused=True);new_names=dict(candidate.named_parameters());new_grad=torch.autograd.grad(lc.mean(),tuple(new_names.values()),allow_unused=True);mapped=dict(zip(new_names,new_grad))
    for name,g in zip(names,old_grad):
        h=mapped[name.replace('readout.','readout.base.',1) if name.startswith('readout.') else name]
        if g is None:assert h is None or torch.count_nonzero(h)==0
        else:
            if g.shape!=h.shape:h=h[:g.shape[0]]
            assert torch.allclose(g,h,rtol=1e-6,atol=1e-8),name
    # These are resource/numerical pilots, not DEV winner selection.
    rows=[]
    for label,model,config in [('parent',copy.deepcopy(parent),args),('grown',candidate,grown_args)]:
        model.train();optimizer=acquired_receiver_optimizer(parent,dense,model);optimizer.zero_grad();state=clone_state(saved['state']);gen=torch.Generator().set_state(saved['generator']);teacher=torch.Generator().set_state(saved['teacher_generator']);cursor=saved['cursor'];tokens=driver.BlockLaneTokens(train,args.train_tokens,args.lanes,args.exposure,args.steps+a.updates,args.chunk);tokens.block=saved['training_block'];curve=[];work=[];begin=time.monotonic()
        for offset in range(a.updates):
            if cursor>=tokens.shape[1]-1:cursor=0;state=None;tokens.advance()
            n=min(args.chunk,tokens.shape[1]-1-cursor);buf=tokens[:,cursor:cursor+n+1];optimizer.zero_grad();loss,state,audits=raw_step(model,state,gen,teacher,config,buf,saved['step']+offset+1)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step();optimizer.zero_grad();cursor+=n;work.extend(audits)
            if offset in (0,1,a.updates-1):curve.append(dict(update=offset+1,train_nll=float(loss.mean()),correction_norm=float(model.readout.correction.weight.norm()) if label=='grown' else 0.))
        prefix=ROOT/f'experiments/results/token_language/{a.tag}_{label}';owned_rng={name:getattr(model,name).get_state() for name in ('site_generator','position_generator','local_generator','alternative_generator')}
        if label=='grown':owned_rng.update({name:getattr(model,name).get_state() for name in ('new_teacher_generator','new_site_generator','new_alternative_generator','new_position_generator')})
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,generator=gen.get_state(),teacher_generator=teacher.get_state(),owned_rng=owned_rng,torch_rng=torch.get_rng_state(),cursor=cursor,training_block=tokens.block,step=saved['step']+a.updates,args=vars(config)),prefix.with_suffix('.pt'))
        rows.append(dict(arm=label,updates=a.updates,curve=curve,wall_s=time.monotonic()-begin,teacher_work=work,final_weights=str(prefix.with_suffix('.pt').relative_to(ROOT)),parameters=sum(p.numel() for p in model.parameters())))
    names=['experiments/credit/protected_depth_learning_pilot.py','sleeping_machines/protected_depth_language.py','sleeping_machines/protected_temporal_branch.py','sleeping_machines/learning/dense_to_receiver_moments.py','sleeping_machines/optional_temporal_depth.py','experiments/multidepth_readout_token_language.py']
    record=dict(status='completed',battle='R1',strong_parent_initial_loss_gradient_context_rng_containment=True,parent='aws_multidepth_readout_8m_20261013_s6_v1',rows=rows,source_sha256={n:driver.sha(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Strongest completed native readout parent, acquired2→protected4stages, same64updateTRAIN continuation numerical/resource pilot. Globalclip1 diagnostic only; calibratedlocal integration and recovery gate before fullfit. Oldperstage teacher opportunity retained and newstream charged; cold newoutputmap has zero internalpredictiongrad initially, learns through live output credit first. No DEVquality win or physical async claim.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
