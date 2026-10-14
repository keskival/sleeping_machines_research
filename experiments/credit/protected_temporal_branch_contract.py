"""R1 protected old feedback while new native state evolves and learns."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import bernoulli_teacher_token_language as driver
from sleeping_machines.optional_temporal_depth import grow_native_core
from sleeping_machines.protected_temporal_branch import protected_branch_features
from sleeping_machines.ordered_token_episodes import token_features
from sleeping_machines.ordered_token_lookup import ordered_content
from sleeping_machines.sparse_counterfactual_episodes import detach

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);start=time.monotonic();base=ROOT/'experiments/results/token_language/aws_integrated_receiver_d8_8m_20261014_s6_v1';r=json.loads(base.with_suffix('.json').read_text());args=SimpleNamespace(**r['args']);tr=driver.load_tokens(args.train_file);counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True);torch.manual_seed(args.seed);parent=driver.Model(args,order,counts);ck=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(ck['model']);bigger=SimpleNamespace(**vars(args));bigger.depth=args.depth+2
 with torch.random.fork_rng():torch.manual_seed(501);fresh=driver.Model(bigger,order,counts)
 core=grow_native_core(parent.core,fresh.core);ids=driver.LaneTokens(tr,262144,392,8)[:,:];gp=torch.Generator().manual_seed(92);gg=torch.Generator().manual_seed(92);sp=sg=None;latency=[]
 for begin in (0,16):
  inputs=ids[:,begin:begin+16];labels=ids[:,begin+1:begin+17]
  xp,sp,_=token_features(parent.core,inputs,sp,gp,eos=None,route_credit='none',token_content=ordered_content(parent.core.content.weight,inputs))
  xg,sg,_=protected_branch_features(core,inputs,sg,gg,eos=None,route_credit='none',token_content=ordered_content(core.content.weight,inputs),output='parent')
  assert torch.equal(xp,xg),'Parent features changed by live side branch'
  assert torch.equal(gp.get_state(),gg.get_state()),'Parent race stream changed'
  for field in ('ctx_vals','ctx_arr','position','has_ctx'):assert torch.equal(sp[field],sg[field]),field
  for field in ('mem','arr','seen'):assert all(torch.equal(u,v) for u,v in zip(sp[field],sg[field][:args.depth])),field
  assert all(torch.isfinite(sg[field]).all() for field in ('branch_ready','query_arrivals'))
  assert (sg['branch_ready']>=sg['ctx_arr'].max(-1).values).all()
  if begin==0:
   lp=parent.readout.nll(xp,labels).mean();lg=parent.readout.nll(xg,labels).mean();pg=torch.autograd.grad(lp,tuple(parent.core.parameters()),allow_unused=True);ng=torch.autograd.grad(lg,tuple(core.parameters()),allow_unused=True)
   old_grad=dict(zip(dict(parent.core.named_parameters()),pg));new_grad=dict(zip(dict(core.named_parameters()),ng))
   for name,g in old_grad.items():
    h=new_grad[name]
    if g is None:assert h is None or torch.count_nonzero(h)==0
    else:
     if h.shape!=g.shape:h=h[:g.shape[0]]
     assert torch.allclose(g,h,rtol=1e-6,atol=1e-8)
  latency.append(float((sg['branch_ready']-sg['ctx_arr'].max(-1).values).mean().detach()));sp=detach(sp);sg=detach(sg)
 assert any(torch.count_nonzero(m)>0 for m in sg['mem'][args.depth:]),'New private memory failed to evolve'
 # One branch-service query uses its actual later availability; parent feedback
 # remains the old state's even when branch features are returned.
 inputs=ids[:,32:48];labels=ids[:,33:49];features,bs,_=protected_branch_features(core,inputs,sg,torch.Generator().manual_seed(93),eos=None,route_credit='none',token_content=ordered_content(core.content.weight,inputs),output='branch')
 loss=parent.readout.nll(features,labels).mean();gradient=torch.autograd.grad(loss,tuple(core.banks.values()),allow_unused=True);credit={k:float(g[args.depth:].norm()) if g is not None else 0. for k,g in zip(core.banks,gradient)}
 assert credit['input']>0 and credit['output']>0 and credit['clock_bias']>0
 names=['experiments/credit/protected_temporal_branch_contract.py','sleeping_machines/protected_temporal_branch.py','sleeping_machines/optional_temporal_depth.py','sleeping_machines/sparse_counterfactual_episodes.py','experiments/bernoulli_teacher_token_language.py']
 record=dict(status='completed',battle='R1',parent_features_private_state_context_timing_rng_preserved_with_live_branch=True,old_factual_gradient_parity=True,new_private_state_evolves=True,branch_later_availability_mean=latency,branch_service_internal_credit=credit,source_sha256={n:driver.sha(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='AcquiredD8 plus native2stage sidebranch,384TRAIN targets/48history; eagercausal eventtime, no physicalscheduler or fit. Parent features served atparenttime, branch features atlateractualsimulatedavailability, backlog local. Supportedselectionteacher and localcaps pending; no latency/speed/quality claim.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
