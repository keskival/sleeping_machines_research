"""B1/R1 identical-label shared-prefix pilot against saved receiver critic."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import unseen_receiver_credit as native
from shared_prefix_receiver_credit import SharedPrefixCredit
ROOT=native.ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','pilot'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic();source=ROOT/'experiments/results/credit/curie_unseen_receiver_v2_20261010T2140Z_pilot.json';old=json.loads(source.read_text());cp=ROOT/old['metrics']['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==old['metrics']['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);model=native.actor(4,170);model.load_state_dict(saved['actor']);small=a.mode=='smoke';n=4 if small else 128;nd=4 if small else 64;nt=nr=2 if small else 8;nr=2 if small else 16;steps=2 if small else 256
    with torch.no_grad(),FlopCounterMode(display=False) as teacher:
        x,y,_=native.collect(model,170,n,nt,list(range(6)));dx,dy,records=native.collect(model,170,nd,nr,list(range(8)),True)
    # Prefix components0:30 are exactly shared across receiver packets.
    assert torch.equal(x[:,:,:,:30],x[:,:1,:,:30].expand(-1,6,-1,-1));assert torch.equal(dx[:,:,:,:30],dx[:,:1,:,:30].expand(-1,8,-1,-1))
    scale=float(y.square().mean().sqrt().clamp_min(1e-8));torch.manual_seed(170);critic=SharedPrefixCredit(model.M,saved['query_bound']).double();opt=torch.optim.AdamW(critic.parameters(),lr=.003,weight_decay=.001);actions=native.base.actions()[None,None].expand(n,6,-1,-1);fit_start=time.monotonic()
    with FlopCounterMode(display=False) as fitting:
        for step in range(steps):
            pred=critic.values(critic.encode_context(x[:,0,:,:30],0,step),x[:,:,:,30:],actions,0,step);loss=((pred-y)/scale).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    fit_wall=time.monotonic()-fit_start
    reference=native.prior.KernelCritic(model.M,saved['query_bound']).double();reference.encoder=nn.GRU(38,32,batch_first=True).double();reference.load_state_dict(saved['models']['structured']);da=native.base.actions()[None,None].expand(nd,8,-1,-1);rows=[];rng=np.random.default_rng(170);indices=rng.integers(0,nd,(2000,nd));fixed=old['metrics']['fixed_train_receiver']*9+old['metrics']['fixed_train_action'];active=torch.tensor([int(m[0,3]) for _,m,_ in records]);mask=(torch.arange(8)[None]>=6)&(torch.arange(8)[None]!=active[:,None])
    with torch.no_grad():
        for name in ['saved_receiver_gru','shared_prefix']:
            tick=time.monotonic()
            with FlopCounterMode(display=False) as query:
                if name=='shared_prefix':p=critic.values(critic.encode_context(dx[:,0,:,:30],0,steps),dx[:,:,:,30:],da,0,steps)
                else:p=reference.values(dx.reshape(nd*8,4,38),da.reshape(nd*8,9,4)).reshape(nd,8,9)
            query_wall=time.monotonic()-tick;chosen=p.flatten(1).argmin(-1);truth=dy.flatten(1);v=truth.gather(1,chosen[:,None]).squeeze(1);gain=(truth[:,fixed]-v).numpy();restricted=p[:,:6].flatten(1).argmin(-1);extra=(truth.gather(1,restricted[:,None]).squeeze(1)-v).numpy();parity=0.
            for i,(ft,fm,first_truth) in enumerate(records[:2]):
                before,_=native.losses(model,ft,fm);r=chosen[i:i+1]//9;j=int(chosen[i]%9);changed,_=native.losses(model,ft,fm,3,r,native.base.actions()[j:j+1]);parity=max(parity,abs(float((changed[:,3:]-before[:,3:]).mean()-first_truth[int(r),j])))
            assert parity<1e-10
            rows.append(dict(arm=name,parameters=sum(z.numel() for z in (critic if name=='shared_prefix' else reference).parameters()),unseen_inactive_relative_mse=float((p[mask]-dy[mask]).square().sum()/dy[mask].square().sum()),gain_vs_fixed=float(gain.mean()),gain_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),expanded_pool_gain=float(extra.mean()),query_supported_flops=query.get_total_flops(),query_wall_s=query_wall,native_write_error=parity))
    checkpoint=out.with_suffix('.pt');torch.save(dict(actor=model.state_dict(),critic=critic.state_dict(),optimizer=opt.state_dict(),args=vars(a),steps=steps),checkpoint)
    sources=['experiments/credit/fit_shared_receiver_credit.py','experiments/credit/shared_prefix_receiver_credit.py',*native.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,args=vars(a),rows=rows,train_prefixes=n,dev_prefixes=nd,train_rollouts=nt,dev_rollouts=nr,steps=steps,fit_wall_s=fit_wall,fit_supported_flops=fitting.get_total_flops(),teacher_supported_flops=teacher.get_total_flops(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),reference_result=str(source.relative_to(ROOT)),reference_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='Pilot: same seed170 actor, exact TRAIN/DEV prefixes, labels, steps, optimizer and nearly matched parameters as saved structured receiver critic. New factorized initialization, no matched encoder initialization. Smoke has fewer labels and is execution only. Frozen synthetic actor, no teacher labels for receivers6/7. Query work excludes common native descriptor generation, reported separately by sparse query diagnostic. New teacher and fitting work plus saved prior fitting charged; no full learning-efficiency or TEST claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
