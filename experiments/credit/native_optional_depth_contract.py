"""R1 actual native timing/state containment and live hypothetical depth credit."""
import argparse,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import bernoulli_teacher_token_language as driver
from sleeping_machines.optional_temporal_depth import grow_native_core,optional_depth_features
from sleeping_machines.ordered_token_lookup import ordered_content
from sleeping_machines.ordered_token_episodes import token_features

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
 base=ROOT/'experiments/results/token_language/aws_integrated_receiver_d8_8m_20261014_s6_v1';r=json.loads(base.with_suffix('.json').read_text());args=SimpleNamespace(**r['args'])
 tr=driver.load_tokens(args.train_file);counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True);torch.manual_seed(args.seed);parent=driver.Model(args,order,counts);ck=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(ck['model']);del ck
 grown_args=SimpleNamespace(**vars(args));grown_args.depth=args.depth+2
 with torch.random.fork_rng():
  torch.manual_seed(501);fresh=driver.Model(grown_args,order,counts)
 before=torch.get_rng_state();core=grow_native_core(parent.core,fresh.core);assert torch.equal(before,torch.get_rng_state())
 ids=driver.LaneTokens(tr,262144,136,8)[:,:];inputs=ids[:,:16];targets=ids[:,1:17]
 def run(c,fn,g,st=None):return fn(c,inputs,st,g,eos=None,route_credit='none',token_content=ordered_content(c.content.weight,inputs))
 gp=torch.Generator().manual_seed(92);gg=torch.Generator().manual_seed(92)
 x,st,_=run(parent.core,token_features,gp);y,gs,_=run(core,optional_depth_features,gg)
 assert torch.equal(x,y),'Prediction-feature containment'
 for field in ('ctx_vals','ctx_arr','position','has_ctx'):assert torch.equal(st[field],gs[field]),field
 for field in ('mem','arr','seen'):
  assert all(torch.equal(u,v) for u,v in zip(st[field],gs[field][:args.depth])),field
 assert torch.equal(gp.get_state(),gg.get_state()),'Parent race RNG changed'
 loss=parent.readout.nll(x,targets).mean();other=parent.readout.nll(y,targets).mean()
 pg=torch.autograd.grad(loss,tuple(parent.core.parameters()),allow_unused=True);qg=torch.autograd.grad(other,tuple(core.parameters()),allow_unused=True)
 old_names=dict(parent.core.named_parameters());new_names=dict(core.named_parameters());old_gradient=dict(zip(old_names,pg));new_gradient=dict(zip(new_names,qg))
 for name,g in old_gradient.items():
  h=new_gradient[name]
  if g is None:assert h is None or torch.count_nonzero(h)==0
  else:
   if h.shape!=g.shape:h=h[:g.shape[0]]
   assert torch.allclose(g,h,rtol=1e-6,atol=1e-8),name
 # Activated alternative uses genuinely new temporal stages, not an output
 # shortcut. Its loss supplies internal stage gradients at a neutral actor.
 core.execution_depth=core.depth
 xx,active_state,_=run(core,optional_depth_features,torch.Generator().manual_seed(92))
 probe_loss=parent.readout.nll(xx,targets).mean();grads=torch.autograd.grad(probe_loss,tuple(core.banks.values()),allow_unused=True)
 added={name:float(g[args.depth:].norm()) if g is not None else 0. for name,g in zip(core.banks,grads)}
 assert added['input']>0 and added['output']>0 and added['gate_w']>0 and added['clock_bias']>0,'New stage internal credit missing'
 names=['experiments/credit/native_optional_depth_contract.py','sleeping_machines/optional_temporal_depth.py','sleeping_machines/stage_gain_token_core.py','sleeping_machines/sparse_counterfactual_episodes.py','sleeping_machines/ordered_token_episodes.py','experiments/bernoulli_teacher_token_language.py']
 record=dict(status='completed',battle='R1',parent_depth=args.depth,allocated_depth=core.depth,neutral_feature_state_timing_rng_gradient_parity=True,activated_alternative_internal_gradient_norm=added,parent_loss=float(loss.detach()),alternative_loss=float(probe_loss.detach()),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={n:driver.sha(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},scope='Acquired actual nativeD8→allocatedD10,128TRAINtargets; no optimizer updates/fit, bank moments/versions not migrated, activation selection/credit policy not integrated. Conditional alternate-path internal gradients prove learnability of that path, not useful depth or asynchronous execution gain.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
