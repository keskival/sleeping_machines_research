"""R1 native D8 future-world cloning, nested replay and own-update contract."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch
from finite_horizon_receipt import HorizonReceipt
from native_horizon_credit_audit import record_native_horizon
from receiver_allocated_future_world import ReceiverAllocatedFutureWorld as NativeTokenFutureWorld, native, ROOT
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW


def fingerprint(world):
    return native.learner_fingerprint((world.model.state_dict(),world.optimizer.state_dict(),
                                     world.state,world.generator.get_state(),world.teacher_generator.get_state(),
                                     [getattr(world.model,n).get_state() for n in ('site_generator','position_generator','local_generator','alternative_generator')],world.next_step,world.allocator.state_dict()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/f'experiments/results/credit/{a.tag}.json'
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(19861)
    args=SimpleNamespace(seed=6,payload=8,depth=8,heads=2,pool=4,input_init='balanced',
        tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
        credit_window=4,state_mode='carry',route_credit='sampled',future_site='uniform',
        free_bias=0.,temperature=1.,compiled=False,lanes=1,teacher_probability=.25,future_score_positions=4,balance=0.,future_scale=1.)
    model=native.Model(args,torch.arange(50257),torch.ones(50257,dtype=torch.long))
    opt=ReceiverBankAdamW(model,lr=.003,weight_decay=.01)
    generator=torch.Generator().manual_seed(19862)
    tape=torch.randint(0,31,(1,49),generator=torch.Generator().manual_seed(19863))
    # Nonzero decoder weights make route effects visible without a preliminary fit.
    with torch.no_grad():
        for p in model.readout.parameters():
            if p.ndim>1:p.add_(.01*torch.randn_like(p))
    issued=native.learner_fingerprint((model.state_dict(),opt.state_dict(),generator.get_state()))
    global_rng=torch.get_rng_state().clone()
    full=NativeTokenFutureWorld(model,opt,None,generator,args)
    split=NativeTokenFutureWorld(model,opt,None,generator,args)
    alt=NativeTokenFutureWorld(model,opt,None,generator,args)
    assert full.optimizer.model is full.model
    full_losses=[];split_losses=[];alt_losses=[];prefix_errors=[]
    for step in range(12):
        x=tape[:,step*4:step*4+4];y=tape[:,step*4+1:step*4+5]
        full_losses.append(full.advance(x,y))
        action=(full.state['race_winners'][0,7,:,0]+1)%4 if step==0 else None
        if step==0:first_action=action.clone()
        alt_losses.append(alt.advance(x,y,force_site=(0,7,0,action) if step==0 else None))
    for step in range(12):
        x=tape[:,step*4:step*4+4];y=tape[:,step*4+1:step*4+5]
        split_losses.append(split.advance(x,y))
        assert torch.equal(split_losses[-1],full_losses[step])
        prefix_errors.append(float((split_losses[-1]-full_losses[step]).abs().max()))
        # Serialize the numerical cut by cloning the learner after the first prefix.
        if step==0:split=NativeTokenFutureWorld(split.model,split.optimizer,split.state,split.generator,args,teacher_generator=split.teacher_generator,next_step=split.next_step,allocator=split.allocator)
    assert fingerprint(full)==fingerprint(split),'Resumed own learner differs'
    assert torch.equal(full.generator.get_state(),alt.generator.get_state()),'Paired race tape differs'
    assert issued==native.learner_fingerprint((model.state_dict(),opt.state_dict(),generator.get_state()))
    assert torch.equal(global_rng,torch.get_rng_state()),'Future worlds consumed global RNG'
    # Later own moments respond to the initial route intervention, not reference gradients.
    moment_gap=sum(float((a['exp_avg']-b['exp_avg']).square().sum())
        for name,opt1 in full.optimizer.receiver.local.optimizers.items()
        for a,b in zip(opt1.state.values(),alt.optimizer.receiver.local.optimizers[name].state.values()))**.5
    assert moment_gap>0,'Intervention did not change future own learning'
    assert full.teacher_work[0]>0 and alt.teacher_work[0]>0,'No alternative teaching exercised'
    ids=dict(objective_id='sum-lane-mean-prequential',world_id='paired-native-production',learner_version='issued0')
    receipt=HorizonReceipt((16,32,48),(.02,-.01,.03),(.2,.3,.5),**ids)
    chunks=[(tape[:,i*4:i*4+4],tape[:,i*4+1:i*4+5]) for i in range(12)]
    audits=[]
    for level in range(3):
        ref=NativeTokenFutureWorld(model,opt,None,generator,args)
        other=NativeTokenFutureWorld(model,opt,None,generator,args)
        audits.append(record_native_horizon(ref,other,receipt,level,(1.,)*48,chunks,initial_force=(0,7,0,first_action)))
    true_terminal=sum(float((r-s).sum()) for r,s in zip(full_losses,alt_losses))
    corrected_mean=sum(q*r['target'] for q,r in zip(receipt.probabilities,audits))
    horizon_mean_error=abs(corrected_mean-true_terminal)
    assert horizon_mean_error<1e-12
    for level,row in enumerate(audits):
        expected=sum(float((r-s).sum()) for r,s in zip(full_losses[:receipt.horizons[level]//4],alt_losses))
        assert abs(row['upper_value']-expected)<1e-12
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and p.suffix=='.py' and p.relative_to(ROOT).parts[0] in ('experiments','sleeping_machines'):
                sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',tag=a.tag,gate='PASS',depth=8,
        prefix_loss_errors=prefix_errors,final_replay_fingerprint_equal=True,
        source_and_global_rng_unchanged=True,paired_race_rng_equal=True,
        future_own_moment_gap=moment_gap,paired_loss_differences=[float((r-s).sum()) for r,s in zip(full_losses,alt_losses)],
        updates_per_full_world=12,total_executed_updates=84,total_scored_targets=336,total_own_gradient_targets=336,
        native_horizon_audits=audits,terminal_advantage=true_terminal,corrected_terminal_mean=corrected_mean,horizon_mean_error=horizon_mean_error,
        teacher_work_full_world=full.teacher_work,teacher_work_alternate_world=alt.teacher_work,
        wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sources,scope='Synthetic legal GPT2 token IDs, native D8/payload8, source-extracted integrated probability0.25 allocated alternative teaching, balance0 and globally clipped receiver-local AdamW. Reconstructed optimizer views and independent moments; actual nested horizons. One initial hard-route intervention; no critic fit, corpus quality or public score.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
