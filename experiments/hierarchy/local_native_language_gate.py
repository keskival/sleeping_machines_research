"""R1 real GPT-2 token/native8depth/local-credit resource gate, not a fit verdict."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM,word_classes


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(161016);began=time.monotonic()
 path=ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin';data=load_fineweb(path)
 counts=np.bincount(np.asarray(data[:4194304]).astype(np.int64),minlength=50257);cls,words=word_classes(counts,256)
 model=LocalNativeDeepLM(50257,cls,words,width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);tokens=torch.from_numpy(np.asarray(data[:33]).astype(np.int64))
 g=torch.Generator().manual_seed(13);alt=torch.Generator().manual_seed(14);state=model.new_state();start=time.monotonic()
 nll,objective,state,packets,report=model.nll(tokens[:-1],tokens[1:],state,g,alt,stats=True)
 forward_s=time.monotonic()-start;t=time.monotonic();objective.backward();backward_s=time.monotonic()-t
 t=time.monotonic();model.deliver_credit(packets,state);local_credit_s=time.monotonic()-t
 finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters());assert finite
 gradient=[]
 for layer in model.core.units:
  gradient.append(sum(float(p.grad.square().sum()) for p in layer.parameters() if p.grad is not None)**.5)
 assert min(gradient)>0
 t=time.monotonic();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();optimizer_s=time.monotonic()-t;elapsed=time.monotonic()-start
 source={str(Path(m.__file__).resolve().relative_to(ROOT)) for name,m in tuple(sys.modules.items()) if (name.startswith('sleeping_machines') or name in ['r1_cerl','r1_local_native_deep','r1_deep_keyed_lm']) and getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
 source.add(str(Path(__file__).relative_to(ROOT)));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 result=dict(status='completed',battle='R1',stage='local_native_deep_language_gate',gate='PASS',targets=32,depth=8,parameters=sum(p.numel() for p in model.parameters()),report=report,layer_gradient_norms=gradient,optimizer_step_s=elapsed,seconds_per_target=elapsed/32,timing=dict(forward_s=forward_s,backward_s=backward_s,local_credit_s=local_credit_s,optimizer_s=optimizer_s),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,whole_wall_s=time.monotonic()-began,source_sha256={n:sha(ROOT/n) for n in sorted(source)},data=dict(file=str(path.relative_to(ROOT)),class_count_prefix_tokens=4194304,scored_prefix_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()),scope='One32targetTRAIN optimizer gate only;realGPT2FineWeb,8native layers,zeroSSM/attention/MLP backbone layers,own-writeeligibility and sampled local counterfactualroutecredit.No recurrentBPTT,no DEV/TEST or quality/efficiency claim.Current episodic pointers cover this supplied credit lane;cross-batch indexed history and asynchronous optimizer/versioning remain acquisition gaps.')
 target=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','optimizer_step_s','seconds_per_target','max_rss_kb','parameters']}),flush=True)
if __name__=='__main__':main()
