"""B4 fixed ridge logit-distillation diagnostic; TRAIN-only, no TEST interface."""
import argparse,hashlib,json,random,resource
import time as timer
from pathlib import Path
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP as Stable
from race_tpp_logspread_marks import RaceTPP as Marks
from single_mark_readout import distributions,SingleMarkReadout
from nonlinear_single_mark_readout import NonlinearSingleMarkReadout
ROOT=adapter.ROOT

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--observed-weight',type=float,choices=[0.,.1],required=True);ap.add_argument('--seed',type=int,choices=[0,1,2],default=0);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=timer.monotonic()
    decision_path=ROOT/'experiments/results/tpp_b4/aws_b4_calibration_confirmation_20261011_decision.json';decision=json.loads(decision_path.read_text());assert decision['status']=='completed' and decision['gate']
    for source,h in decision['source_sha256'].items():assert digest(ROOT/source)==h
    tags=['aws_b4_logspread_develop_20261010T2112Z','aws_b4_marks_develop_20261010T2300Z'] if a.seed==0 else [f'aws_b4_repair_confirm_20261011_s{a.seed}',f'aws_b4_marks_confirm_20261011_s{a.seed}']
    records=[json.loads((ROOT/'experiments/results/tpp_b4'/f'{tag}.json').read_text()) for tag in tags]
    for r in records:
        assert r['status']=='completed' and r['seed']==a.seed
        for source,h in r['source_sha256'].items():assert digest(ROOT/source)==h
        for part,h in r['data_sha256'].items():assert digest(adapter.DATA/'github_filtered'/'split_1'/f'{part}.json')==h
    row=decision['rows'][a.seed];alpha=row['alpha'];train,K=adapter.load('github_filtered',1,'train');dev,_=adapter.load('github_filtered',1,'val');cfg=records[0]['config']
    pilot=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4_logspread_pilot_20261010T2003Z.json').read_text())
    gaps=np.concatenate([np.diff(t) for t,m in train]);pos=gaps[gaps>0];qs=np.log(np.quantile(pos,np.linspace(.1,.9,cfg['n_lognormal']))).tolist()
    args=(K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],pilot['scale'],qs,pilot['recording_cell'],0,None,cfg['state_modes'])
    models=[Stable(*args).eval(),Marks(*args).eval()];checkpoints=[ROOT/r['best_checkpoint'] for r in records]
    for model,path in zip(models,checkpoints):model.load_state_dict(torch.load(path,weights_only=True))
    frozen={k:v.detach().clone() for k,v in models[0].state_dict().items()};cache=[];tick=timer.monotonic()
    # Cache TRAIN first; no DEV values take part in fitting.
    for seq in train:
        t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)))
        time,b,x=distributions(models[0],t,m,mask);_,j,_=distributions(models[1],t,m,mask)
        teacher=torch.logaddexp(b+np.log1p(-alpha),j+np.log(alpha));valid=mask[:,1:]
        assert torch.isfinite(teacher[valid]).all() and torch.allclose(teacher[valid].exp().sum(-1),torch.ones_like(time[valid]),atol=1e-10)
        cache.append((x[valid].numpy(),(teacher-b)[valid].numpy(),b[valid].numpy(),teacher[valid].numpy()))
    preparation_s=timer.monotonic()-tick
    x=np.concatenate([v[0] for v in cache]);y=np.concatenate([v[1] for v in cache]);mean=x.mean(0);std=x.std(0).clip(min=1e-8)
    design=np.column_stack([(x-mean)/std,np.ones(len(x))]);y=y-y.mean(1,keepdims=True)
    # Fixed lambda=1, squared centered-logit objective; no DEV grid/selection.
    tick=timer.monotonic();penalty=np.eye(design.shape[1]);penalty[-1,-1]=0
    coeff=np.linalg.solve(design.T@design+penalty,design.T@y)
    design_t=torch.tensor(design);offset=torch.tensor(np.concatenate([v[2] for v in cache]));teacher_prob=torch.tensor(np.concatenate([v[3] for v in cache])).exp()
    labels=torch.tensor(np.concatenate([marks[1:] for times,marks in train]),dtype=torch.long);assert len(labels)==len(x)
    teacher_prob=(1-a.observed_weight)*teacher_prob+a.observed_weight*torch.nn.functional.one_hot(labels,num_classes=models[0].K).to(teacher_prob.dtype)
    weights=torch.tensor(coeff,requires_grad=True);opt=torch.optim.LBFGS([weights],lr=1.,max_iter=100,tolerance_grad=1e-10,tolerance_change=1e-12,line_search_fn='strong_wolfe');closure_calls=0
    def closure():
        nonlocal closure_calls
        opt.zero_grad();lp=torch.log_softmax(offset+design_t@weights,-1)
        objective=-(teacher_prob*lp).sum()/len(x)+.5*weights[:-1].square().sum()/len(x)
        objective.backward();closure_calls+=1;return objective
    initial_objective=float(closure().detach());opt.step(closure);final_objective=float(closure().detach());maximum_objective_gradient=float(weights.grad.abs().max());coeff=weights.detach().numpy();fit_s=timer.monotonic()-tick
    torch.manual_seed(a.seed)
    student=NonlinearSingleMarkReadout(models[0],torch.tensor(mean),torch.tensor(std),torch.tensor(coeff)).eval()
    fixed_logits=offset+design_t@weights.detach();normalized=torch.tensor((x-mean)/std)
    def score(model,seqs):
        losses=[];tick=timer.monotonic();targets=0
        for seq in seqs:
            t,m,mask=next(adapter.base.batches([seq],1,False,random.Random(0)));valid=mask[:,1:]
            if isinstance(model,SingleMarkReadout):time,lp=model.event_distributions(t,m,mask)
            else:time,lp,_=distributions(model,t,m,mask)
            mark=lp.gather(-1,m[:,1:,None]).squeeze(-1);losses.append((-float(time[valid].sum()),-float(mark[valid].sum())));targets+=int(valid.sum())
        lt,lm=np.mean(losses,0);return dict(L_T=float(lt),L_M=float(lm),total=float(lt+lm),wall_s=timer.monotonic()-tick,targets=targets,sequences=len(losses))
    residual_opt=torch.optim.AdamW(student.residual.parameters(),lr=.003,weight_decay=1e-4);generator=torch.Generator().manual_seed(a.seed);training_history=[];updates=0;nonlinear_fit_s=0.;curve=[];evaluation_curve_s=0.
    target_path=ROOT/'experiments/results/tpp_b4/aws_single_readout_decision_20261011.json';target_record=json.loads(target_path.read_text());assert target_record['status']=='completed' and target_record['gate'];target=target_record['rows'][a.seed]['student']['L_M']
    initial=score(student,dev);evaluation_curve_s+=initial['wall_s'];curve.append(dict(passes=0,optimizer_updates=0,unique_train_targets=len(x),target_presentations=0,dev=initial,readout_fitting_s=fit_s,teacher_preparation_s=preparation_s,cumulative_work_wall_s=preparation_s+fit_s,prerequisite_fitting_s=sum(r['fitting_wall_s'] for r in records),reached=initial['L_M']<=target+1e-9))
    for epoch in range(50):
        epoch_start=timer.monotonic()
        order=torch.randperm(len(x),generator=generator);total=0.
        for indices in order.split(512):
            residual_opt.zero_grad();lp=torch.log_softmax(fixed_logits[indices]+student.residual(normalized[indices]),-1)
            objective=-(teacher_prob[indices]*lp).sum()/len(indices);objective.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.residual.parameters())
            residual_opt.step();updates+=1;total+=float(objective.detach())*len(indices)
        nonlinear_fit_s+=timer.monotonic()-epoch_start
        training_history.append(dict(epoch=epoch,teacher_crossentropy=total/len(x)))
        if epoch+1 in [5,10,20,35,50]:
            point=score(student,dev);evaluation_curve_s+=point['wall_s'];curve.append(dict(passes=epoch+1,optimizer_updates=updates,unique_train_targets=len(x),target_presentations=(epoch+1)*len(x),dev=point,readout_fitting_s=fit_s+nonlinear_fit_s,teacher_preparation_s=preparation_s,cumulative_work_wall_s=preparation_s+fit_s+nonlinear_fit_s,prerequisite_fitting_s=sum(r['fitting_wall_s'] for r in records),reached=point['L_M']<=target+1e-9))
    baseline=score(models[0],dev);actual=score(student,dev);assert abs(baseline['L_M']-row['baseline']['L_M'])<1e-9 and abs(actual['L_T']-baseline['L_T'])<1e-12
    for k,v in student.core.state_dict().items():assert torch.equal(v,frozen[k])
    t,m,mask=next(adapter.base.batches([dev[0]],1,False,random.Random(0)))
    _,lp=student.event_distributions(t,m,mask);changed=m.clone();changed[:,1:]=(changed[:,1:]+1)%K;_,alt=student.event_distributions(t,changed,mask)
    causal_error=float((lp[:,0]-alt[:,0]).abs().max());assert causal_error<1e-12
    # Warm repeated serving comparison; same DEV interface and all full distributions.
    single_times=[];pair_times=[]
    for _ in range(3):
        single_times.append(score(student,dev)['wall_s']);tick=timer.monotonic();score(models[0],dev);score(models[1],dev);pair_times.append(timer.monotonic()-tick)
    out=ROOT/'experiments/results/tpp_b4';checkpoint=out/f'{a.tag}.pt';assert not checkpoint.exists();torch.save(student.state_dict(),checkpoint)
    gain=baseline['L_M']-actual['L_M'];retention=gain/row['gain'];sources=['experiments/tpp/nonlinear_single_mark_readout.py','experiments/tpp/measure_observation_readout_learning_curve.py','experiments/tpp/single_mark_readout.py','experiments/tpp/race_tpp_b4g.py','experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_logspread_marks.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_v20.py']
    result=dict(status='completed',tag=a.tag,battle='B4 development',seed=a.seed,stage='registered_readout_learning_curve',observed_train_weight=a.observed_weight,learning_curve=curve,curve_evaluation_s=evaluation_curve_s,registered_target_mark_loss=target,nonlinear_fit_s=nonlinear_fit_s,training_history=training_history,optimizer_updates=updates,nonlinear_target_presentations=50*len(x),closure_calls=closure_calls,initial_objective=initial_objective,final_objective=final_objective,maximum_objective_gradient=maximum_objective_gradient,baseline_dev=baseline,student_dev=actual,teacher_dev=row['calibrated'],mark_gain=gain,teacher_gain=row['gain'],retained_teacher_gain=retention,gate=gain>=1 and retention>=.8,rule='Fixed TRAIN90% teacher/10% observed-label probability target, KL linear initialization plus64-wide Tanh residual, AdamW0.003/weight_decay1e-4,50 passes/batch512; no DEV selection; gate DEV mark gain>=1 and >=80% of confirmed teacher gain; timing/core exactly fixed, no DEV tuning.',feature_dimension=x.shape[1],training_targets=len(x),teacher_preparation_s=preparation_s,readout_fit_s=fit_s,prerequisite_fitting_s=sum(r['fitting_wall_s'] for r in records),parameters=sum(p.numel() for p in student.parameters()),teacher_parameters=sum(p.numel() for model in models for p in model.parameters())+1,warm_single_dev_s=float(np.median(single_times)),warm_pair_dev_s=float(np.median(pair_times)),first_target_suffix_mark_error=causal_error,source_sha256={s:digest(ROOT/s) for s in sources},checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),input_sha256={str(target_path.relative_to(ROOT)):digest(target_path),str(decision_path.relative_to(ROOT)):digest(decision_path),**{str(p.relative_to(ROOT)):digest(p) for p in checkpoints}},data_sha256=records[0]['data_sha256'],wall_s=timer.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One fixed Github split/seed distillation diagnostic. Student uses baseline causal history representations and candidate elapsed time; no target-mark feature or teacher at serving. Both teacher fits/full mark distributions/preparation charged. No TEST, sparse-route or matched-quality efficiency win unless gate plus work protocol demonstrated.')
    with (out/f'{a.tag}.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
