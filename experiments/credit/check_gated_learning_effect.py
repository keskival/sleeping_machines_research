"""B1/R1 bounds and exact lazy-selection contracts on saved learned correction."""
import argparse,hashlib,json,resource,time
import torch
from torch.utils.flop_counter import FlopCounterMode
from fit_optimizer_aware_credit import Correction,ROOT,native
from gated_learning_effect import bound_packet,select

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();begin=time.monotonic();path=ROOT/'experiments/results/credit/curie_paired_learning_effect_20261011T0030Z_fit.json';source=json.loads(path.read_text());cp=ROOT/source['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);head=Correction(saved['scale']).double();head.load_state_dict(saved['head']);teacher=torch.load(ROOT/saved['source_checkpoint'],map_location='cpu',weights_only=False);dev=teacher['dev_tensors'];z=dev['z'];base=dev['base'];actions=native.base.actions()[None,None].expand(len(z),8,-1,-1)
    with torch.no_grad(),FlopCounterMode(display=False) as counter:
        packet=bound_packet(head,.25,0);full=head.values(z,actions);assert float(full.abs().max())<=packet.value+1e-12;counts=[]
        for alpha in [0,.25,.5,1.]:
            chosen,need=select(base,z,actions,head,alpha,packet,0);reference=(base+alpha*full).flatten(1).argmin(1);assert torch.equal(chosen,reference);counts.append(dict(alpha=alpha,evaluated_prefixes=int(need.sum()),prefixes=len(z)))
        forced=base.clone();forced[0]=0.;forced[0,0,1]=-10.;chosen,need=select(forced,z,actions,head,1.,packet,0);assert not need[0] and torch.equal(chosen,(forced+full).flatten(1).argmin(1))
        rejected=0
        for radius,version in [(.3,0),(.25,1)]:
            aa=actions*(radius/.25)
            try:select(base,z,aa,head,1.,packet,version)
            except ValueError:rejected+=1
        assert rejected==2
    sources=['experiments/credit/check_gated_learning_effect.py','experiments/credit/gated_learning_effect.py','experiments/credit/fit_optimizer_aware_credit.py',*native.SOURCES];result=dict(status='completed',battle='B1/R1',tag=a.tag,metrics=dict(correction_bound=packet.value,observed_max=float(full.abs().max()),selection_mismatch=0,forced_margin_skip=True,stale_or_radius_rejections=rejected,conditional_queries=counts),wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Learned-model bound/selection arithmetic on previously examined saved DEV. No true-loss certificate, fresh utility assessment, training-efficiency or hardware claim. Full baseline work remains.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':main()
