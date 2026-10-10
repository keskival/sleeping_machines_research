"""R1/B1 bounded depth transfer of the denoised future-credit interface.
Identical v3 critics/labels; initialized frozen actors, not deep actor fitting.
"""
import argparse,hashlib,json,resource,time
import torch
from torch.utils.flop_counter import FlopCounterMode
import averaged_future_credit_v3 as prior
SOURCES=['experiments/credit/averaged_future_credit_depth_v4.py',*prior.SOURCES]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','pilot'],required=True);ap.add_argument('--depth',type=int,choices=[4,8],required=True);ap.add_argument('--seed',type=int,default=170);ap.add_argument('--steps',type=int,default=256);ap.add_argument('--train-packets',type=int,default=128);ap.add_argument('--dev-packets',type=int,default=64);ap.add_argument('--train-rollouts',type=int,default=8);ap.add_argument('--dev-rollouts',type=int,default=16);a=ap.parse_args()
    if a.mode=='smoke':a.steps=2;a.train_packets=a.dev_packets=2;a.train_rollouts=a.dev_rollouts=2
    out=prior.base.ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    original=prior.base.actor
    def deeper(depth,seed):return original(a.depth,seed)
    prior.base.actor=deeper
    try:
        with FlopCounterMode(display=False) as counter:metrics=prior.run(a)
    finally:prior.base.actor=original
    metrics['actor_depth']=a.depth;metrics['depth_scope']='Frozen initialized actor differs with depth; same sampled external prefixes/continuations and critic interface. Compare utility and calibration against each teacher-specific TRAIN constant. Not actor-learning efficiency with depth or comparable-quality compute.'
    result=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),flop_scope='All shared teacher generation and four fits included; supported PyTorch formulas exclude special/unsupported arithmetic.',source_sha256={p:hashlib.sha256((prior.base.ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
if __name__=='__main__':main()
