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
from native_token_future_world import NativeTokenFutureWorld, native, ROOT


def fingerprint(world):
    return native.learner_fingerprint((world.model.state_dict(),world.optimizer.state_dict(),
                                     world.state,world.generator.get_state()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/f'experiments/results/credit/{a.tag}.json'
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(19861)
    args=SimpleNamespace(seed=6,payload=8,depth=8,heads=2,pool=4,input_init='balanced',
        tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
        credit_window=4,state_mode='carry',route_credit='sampled',future_site='uniform',
        free_bias=0.,temperature=1.,compiled=False)
    model=native.Model(args,torch.arange(50257),torch.ones(50257,dtype=torch.long))
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(19862)
    tape=torch.randint(0,31,(1,13),generator=torch.Generator().manual_seed(19863))
    # Nonzero decoder weights make route effects visible without a preliminary fit.
    with torch.no_grad():
        for p in model.readout.parameters():
            if p.ndim>1:p.add_(.01*torch.randn_like(p))
    issued=native.learner_fingerprint((model.state_dict(),opt.state_dict(),generator.get_state()))
    global_rng=torch.get_rng_state().clone()
    full=NativeTokenFutureWorld(model,opt,None,generator,args)
    split=NativeTokenFutureWorld(model,opt,None,generator,args)
    alt=NativeTokenFutureWorld(model,opt,None,generator,args)
    assert all(p is q for p,q in zip(full.model.parameters(),full.optimizer.param_groups[0]['params']))
    full_losses=[];split_losses=[];alt_losses=[];prefix_errors=[]
    for step in range(3):
        x=tape[:,step*4:step*4+4];y=tape[:,step*4+1:step*4+5]
        full_losses.append(full.advance(x,y))
        action=(full.state['race_winners'][0,7,:,0]+1)%4 if step==0 else None
        alt_losses.append(alt.advance(x,y,force_site=(0,7,0,action) if step==0 else None))
    for step in range(3):
        x=tape[:,step*4:step*4+4];y=tape[:,step*4+1:step*4+5]
        split_losses.append(split.advance(x,y))
        assert torch.equal(split_losses[-1],full_losses[step])
        prefix_errors.append(float((split_losses[-1]-full_losses[step]).abs().max()))
        # Serialize the numerical cut by cloning the learner after the first prefix.
        if step==0:split=NativeTokenFutureWorld(split.model,split.optimizer,split.state,split.generator,args)
    assert fingerprint(full)==fingerprint(split),'Resumed own learner differs'
    assert torch.equal(full.generator.get_state(),alt.generator.get_state()),'Paired race tape differs'
    assert issued==native.learner_fingerprint((model.state_dict(),opt.state_dict(),generator.get_state()))
    assert torch.equal(global_rng,torch.get_rng_state()),'Future worlds consumed global RNG'
    # Later own moments respond to the initial route intervention, not reference gradients.
    parameter_pairs=list(zip(full.model.parameters(),alt.model.parameters()))
    assert all((p in full.optimizer.state)==(q in alt.optimizer.state) for p,q in parameter_pairs)
    moment_gap=sum(float((full.optimizer.state[p]['exp_avg']-alt.optimizer.state[q]['exp_avg']).square().sum())
                   for p,q in parameter_pairs if p in full.optimizer.state)**.5
    assert moment_gap>0,'Intervention did not change future own learning'
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
        updates_per_full_world=3,total_executed_updates=9,total_scored_targets=36,total_own_gradient_targets=36,
        wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sources,scope='Synthetic legal GPT2 token IDs, native D8/payload8, own factual gradients/clipped AdamW only. One initial hard-route intervention; no production Bernoulli alternative teacher, critic fit, corpus quality or public score.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
