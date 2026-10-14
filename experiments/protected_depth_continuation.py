"""R1 paired native parent/protected hierarchy continuation with fixed local caps.

One predeclared final DEV endpoint; no parameter search or early quality veto.
Full receiver, numerical and random state is saved at bounded recovery cuts.
"""
import argparse,copy,json,platform,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
import protected_depth_local_gate as gate
from public_token_scoring import score
pilot=gate.pilot
GATE='experiments/results/credit/aws_protected_depth_local_gate_20261014_v1.json'

def snapshot(model,opt,state,gen,teacher,cursor,block,step,offset,binding,wall,work):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,generator=gen.get_state(),teacher_generator=teacher.get_state(),owned_rng=gate.private_state(model),torch_rng=torch.get_rng_state(),cursor=cursor,training_block=block,step=step,offset=offset,binding=binding,training_s=wall,teacher_work=work)

def atomic_save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--arm',choices=['parent','grown'],required=True);p.add_argument('--updates',type=int,default=4096);a=p.parse_args();assert a.updates>0;torch.set_num_threads(1);started=time.monotonic();receipt=ROOT/GATE;r=json.loads(receipt.read_text());assert r['status']=='completed'
    for name,digest in {**r['source_sha256'],**r['input_sha256']}.items():assert pilot.driver.sha(ROOT/name)==digest,name
    row=next(x for x in r['rows'] if x['arm']==a.arm);assert row['actual_local_update_full_recovery'] and row['heldout_train_clip_fraction']>0
    model,oldopt,args,train,warm,warm_path=gate.restore_arm(a.arm);opt=gate.CalibratedReceiverBankAdamW(model,row['calibration']['caps'],lr=args.lr,weight_decay=args.weight_decay);opt.load_state_dict({**oldopt.state_dict(),'bank_caps':row['calibration']['caps']});del oldopt
    prefix=ROOT/f'experiments/results/token_language/{a.tag}';progress=prefix.with_suffix('.progress.pt');source=pilot.driver.sha(Path(__file__));binding=dict(tag=a.tag,arm=a.arm,updates=a.updates,driver_sha256=source,gate_sha256=pilot.driver.sha(receipt),warm_sha256=pilot.driver.sha(warm_path))
    ck=warm;offset=0;elapsed=0.;work=dict(old_calls=0,new_calls=0,replayed_targets=0,scored_targets=0)
    if progress.exists():
        ck=torch.load(progress,weights_only=False);assert ck['binding']==binding;model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);offset=ck['offset'];elapsed=ck['training_s'];work=ck['teacher_work']
    for name,rng in ck['owned_rng'].items():getattr(model,name).set_state(rng)
    torch.set_rng_state(ck['torch_rng']);state=pilot.clone_state(ck['state']);gen=torch.Generator().set_state(ck['generator']);teacher=torch.Generator().set_state(ck['teacher_generator']);cursor=ck['cursor'];tokens=pilot.driver.BlockLaneTokens(train,args.train_tokens,args.lanes,args.exposure,args.steps+64+a.updates,args.chunk);tokens.block=ck['training_block'];model.train();segment=time.monotonic();curve=[]
    for index in range(offset,a.updates):
        if cursor>=tokens.shape[1]-1:cursor=0;state=None;tokens.advance()
        n=min(args.chunk,tokens.shape[1]-1-cursor);buf=tokens[:,cursor:cursor+n+1];opt.zero_grad();loss,state,audits=pilot.raw_step(model,state,gen,teacher,args,buf,warm['step']+index+1);opt.step();opt.zero_grad();cursor+=n
        for audit in audits:
            work[audit['kind']+'_calls']+=1;work['replayed_targets']+=audit['replayed_targets'];work['scored_targets']+=audit['scored_targets']
        if (index+1)%256==0 or index+1==a.updates:
            training_s=elapsed+time.monotonic()-segment;value=snapshot(model,opt,state,gen,teacher,cursor,tokens.block,warm['step']+index+1,index+1,binding,training_s,work);atomic_save(value,progress);curve.append(dict(update=index+1,train_nll=float(loss.mean()),training_s=training_s));print(json.dumps(curve[-1]),flush=True)
    final=torch.load(progress,weights_only=False);atomic_save(final,prefix.with_suffix('.pt'));model.load_state_dict(final['model'])
    # Scoring state is independent of the complete saved TRAIN learner.
    if a.arm=='grown':model.core.new_depth_generator.manual_seed(args.seed+100001)
    dev=pilot.driver.load_tokens(args.dev_file);begin=time.monotonic();scored=score(model,dev,args,SimpleNamespace(Model=type(model)),25165824,8192,1024,8);score_s=time.monotonic()-begin
    parent_record=json.loads((ROOT/'experiments/results/token_language/aws_multidepth_readout_8m_20261013_s6_v1.json').read_text());pilot_record=json.loads((ROOT/f'experiments/results/diagnostics/{gate.PILOT}.json').read_text());pilot_row=next(x for x in pilot_record['rows'] if x['arm']==a.arm)
    result=dict(status='completed',battle='R1',arm=a.arm,args=vars(args),binding=binding,evidence_role='driver_smoke' if a.updates<4096 else 'controlled_development_comparison',added_updates=a.updates,added_targets=a.updates*args.chunk*args.lanes,warmup_targets=64*args.chunk*args.lanes,calibration_targets=row['calibration_targets'],acquired_parent_targets=parent_record['presentations_total'],added_training_s=final['training_s'],warmup_wall_s=pilot_row['wall_s'],calibration_gate_total_wall_s=r['wall_s'],scoring_s=score_s,scored=scored,curve=curve,teacher_work=work,parameters=sum(p.numel() for p in model.parameters()),source_sha256={**r['source_sha256'],'experiments/protected_depth_continuation.py':source,'experiments/public_token_scoring.py':pilot.driver.sha(ROOT/'experiments/public_token_scoring.py')},input_sha256={GATE:pilot.driver.sha(receipt),str(warm_path.relative_to(ROOT)):pilot.driver.sha(warm_path)},host=platform.node(),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-started,scope='Final weights, one seed, paired8192 reusedDEVtargets/1024history/noEOSreset. Same acquired parent and64warmup; requested additional TRAIN updates/data/order matched between full comparison arms; each arm calibrated on its actual warmed actor. Grown pays extra native stages and independent supported new-stage teacher, preserving old audit opportunity. Training time/teacher work reported; no iso-FLOP, physical asynchronous speed, public benchmark or isolated depth-only win claim. No selection on this endpoint; current eager implementation always evaluates the branch.')
    with prefix.with_suffix('.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(scored),flush=True)
if __name__=='__main__':main()
