"""B1/R1 reuse paid paired future packets; preserve baseline response."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from fit_optimizer_aware_credit import Correction,ROOT,native

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic();source=ROOT/'experiments/results/credit/curie_optimizer_aware_fit_20261011T0010Z_pilot.json';prior=json.loads(source.read_text());cp=ROOT/prior['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==prior['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);train=saved['train_tensors'];dev=saved['dev_tensors'];target=train['aware']-train['fixed'];scale=float(target.square().mean().sqrt().clamp_min(1e-8));torch.manual_seed(170);head=Correction(scale).double();opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.001);n=len(target);actions=native.base.actions()[None,None].expand(n,6,-1,-1)
    with FlopCounterMode(display=False) as fitting:
        for step in range(256):
            pred=head.values(train['z'],actions);loss=((pred-target)/scale).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    nd=len(dev['base']);indices=np.random.default_rng(170).integers(0,nd,(2000,nd));truth=dev['aware'].flatten(1);fixed=int(train['aware'].mean(0).flatten().argmin());rows=[];values={}
    with torch.no_grad():
        effect=head.values(dev['z'],native.base.actions()[None,None].expand(nd,8,-1,-1));predictions={'saved_frozen_credit':dev['base'],'paired_learning_effect':dev['base']+effect}
        for name,pred in predictions.items():
            chosen=pred.flatten(1).argmin(-1);v=truth.gather(1,chosen[:,None]).squeeze(1);values[name]=v;gain=(truth[:,fixed]-v).numpy();rows.append(dict(arm=name,learning_aware_relative_mse=float((pred-dev['aware']).square().sum()/dev['aware'].square().sum()),unseen_relative_mse=float((pred[:,6:]-dev['aware'][:,6:]).square().sum()/dev['aware'][:,6:].square().sum()),gain_vs_train_fixed=float(gain.mean()),gain_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),gain_vs_noop=float(-v.mean()),chosen_receiver_action=torch.stack((chosen//9,chosen%9),1).tolist()))
        gain=(values['saved_frozen_credit']-values['paired_learning_effect']).numpy();effect_true=dev['aware']-dev['fixed'];effect_ratio=float((effect-effect_true).square().sum()/effect_true.square().sum())
    checkpoint=out.with_suffix('.pt');torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),scale=scale,source_checkpoint=str(cp.relative_to(ROOT)),source_checkpoint_sha256=prior['checkpoint_sha256']),checkpoint);sources=['experiments/credit/fit_paired_learning_effect.py','experiments/credit/fit_optimizer_aware_credit.py','experiments/credit/optimizer_aware_route_credit.py','experiments/credit/shared_prefix_receiver_credit.py',*native.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,rows=rows,gain_over_preserved_baseline=float(gain.mean()),gain_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),learning_effect_relative_mse=effect_ratio,train_scale=scale,fit_supported_flops=fitting.get_total_flops(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),reused_teacher_result=str(source.relative_to(ROOT)),reused_teacher_checkpoint_sha256=prior['checkpoint_sha256'],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='One-seed development revision on previously examined synthetic DEV packets. Same2817-parameter correction, paired learning-effect target and TRAIN-only effect scaling; baseline/representation preserved. All original teacher/baseline/fitting work remains charged; new fit has no new replay. No independent confirmation, online critic co-adaptation, full actor learning, efficiency or TEST claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
