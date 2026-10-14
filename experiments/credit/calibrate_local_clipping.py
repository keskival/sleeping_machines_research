"""R1 same-issued-learner TRAIN calibration; no actor fitting or DEV selection."""
import argparse,copy,json,math,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import integrated_receiver_token_language as driver
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.depth_credit_allocator import DepthCreditAllocator
from sleeping_machines.learning.calibrated_bank_caps import calibrate_bank_caps

def adam_displacement(p,g,opt):
 group=opt.param_groups[0]
 if group.get('amsgrad') or group.get('maximize') or group.get('differentiable'):raise ValueError('Unsupported Adam geometry')
 state=opt.state.get(p,{})
 b1,b2=group['betas'];t=float(state.get('step',0))+1
 m=state.get('exp_avg',torch.zeros_like(p))*b1+g*(1-b1)
 v=state.get('exp_avg_sq',torch.zeros_like(p))*b2+g.square()*(1-b2)
 delta=-group['lr']*(m/(1-b1**t))/(v.sqrt()/math.sqrt(1-b2**t)+group['eps'])-group['lr']*group['weight_decay']*p.detach()
 return float(delta.double().square().sum())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);tag=ap.parse_args().tag;torch.set_num_threads(1);began=time.monotonic()
 run='aws_integrated_receiver_d8_8m_20261014_s6_v1';base=ROOT/f'experiments/results/token_language/{run}';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed'
 for name,digest in r['source_sha256'].items():assert driver.sha(ROOT/name)==digest,name
 args=SimpleNamespace(**r['args']);tr=driver.load_tokens(args.train_file);counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();torch.manual_seed(args.seed);model=driver.Model(args,torch.argsort(counts,descending=True,stable=True),counts)
 ck=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt=ReceiverBankAdamW(model,lr=args.lr,weight_decay=args.weight_decay);opt.load_state_dict(ck['optimizer']);state=driver.clone_numeric_state(ck['state'])
 for name in ('site_generator','position_generator','local_generator','alternative_generator'):getattr(model,name).set_state(ck[name])
 gen=torch.Generator().set_state(ck['generator']);teacher_gen=torch.Generator().set_state(ck['teacher_generator']);allocator=DepthCreditAllocator(args.depth,args.heads,args.credit_window,args.allocation_exploration,args.allocation_decay);allocator.load_state_dict(ck['depth_credit_allocator']);torch.set_rng_state(ck['torch_rng'])
 source=(ROOT/'experiments/integrated_receiver_token_language.py').read_text();start=source.index('        use_future = bool(');end=source.index('        loss.backward()',start)+len('        loss.backward()')
 body='\n'.join(line[4:] for line in source[start:end].splitlines())
 function='def raw_step(model,opt,state,gen,teacher_gen,allocator,a,step,buf):\n    n=buf.shape[1]-1\n    opt.zero_grad(set_to_none=True)\n    teacher_calls=teacher_replay_tokens=teacher_scored_targets=0\n'+body+'\n    return detach(state),(teacher_calls,teacher_replay_tokens,teacher_scored_targets)\n'
 ns=dict(driver.__dict__);exec(compile(function,'source-bound-raw-credit','exec'),ns);kernel=ns['raw_step']
 train=driver.BlockLaneTokens(tr,args.train_tokens,args.lanes,args.exposure,args.steps+32,args.chunk);train.block=ck['training_block'];cursor=ck['cursor'];norms={b:[] for b in opt.receiver.mapping};scales=[];held=[];work=[0,0,0];private_before={n:p.detach().clone() for n,p in model.named_parameters()}
 for index in range(32):
  if cursor>=train.shape[1]-1:cursor=0;state=None;train.advance()
  n=min(args.chunk,train.shape[1]-1-cursor);state,cost=kernel(model,opt,state,gen,teacher_gen,allocator,args,ck['step']+index+1,train[:,cursor:cursor+n+1]);cursor+=n;work=[a+b for a,b in zip(work,cost)]
  packets=list(opt.receiver.packets('calibration'));raw={p.bank:p.gradient for p in packets};total=math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None));scale=min(1.,1/(total+1e-6))
  if index<24:
   scales.append(scale)
   for bank in norms:norms[bank].append(float(raw[bank].double().norm()) if bank in raw else 0.)
  else:held.append((raw,scale,total))
 for n,p in model.named_parameters():assert torch.equal(p,private_before[n]),'Calibration changed actor'
 fitted=calibrate_bank_caps(norms,scales,quantile=.9,budget=1.25);metrics={name:dict(clipped=0,observed=0,gradient_square=0.,adam_displacement_square=0.) for name in ['global1','local1','calibrated1p25']};placements={name:{k:dict(clipped=0,observed=0) for k in set(opt.receiver.placements.values())} for name in metrics}
 for raw,scale,total in held:
  for bank,g in raw.items():
   norm=float(g.double().norm());p=opt.receiver.local.parameters[bank];optim=opt.receiver.local.optimizers[bank]
   for name,factor in [('global1',scale),('local1',min(1.,1/(norm+1e-6))),('calibrated1p25',min(1.,fitted['caps'][bank]/(norm+1e-6)))]:
    bounded=g*factor;m=metrics[name];m['observed']+=1;m['clipped']+=int(factor<1);m['gradient_square']+=float(bounded.double().square().sum());m['adam_displacement_square']+=adam_displacement(p,bounded,optim);place=placements[name][opt.receiver.placements[bank]];place['observed']+=1;place['clipped']+=int(factor<1)
 for m in metrics.values():m['rms_combined_gradient_norm']=(m.pop('gradient_square')/len(held))**.5;m['rms_adam_displacement']=(m.pop('adam_displacement_square')/len(held))**.5;m['activation_fraction']=m['clipped']/m['observed']
 sources=dict(r['source_sha256'])
 for name in ['experiments/credit/calibrate_local_clipping.py','sleeping_machines/learning/calibrated_bank_caps.py']:sources[name]=driver.sha(ROOT/name)
 inputs={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]}
 out=ROOT/f'experiments/results/credit/{tag}.json'
 with out.open('x') as f:json.dump(dict(status='completed',battle='R1',calibration=fitted,heldout_metrics=metrics,placement_activation=placements,teacher_work=work,calibration_chunks=24,heldout_chunks=8,source_sha256=sources,input_sha256=inputs,wall_s=time.monotonic()-began,scope='Frozen completed actor/moments;32subsequent TRAIN chunks with source-bound raw allocatedteacher, causal numericalstate/allocator/RNG evolve but no actor updates. Same raw gradient per policy; fixed existing Adam moments predict one-step displacement, not actual continued-fit quality. No DEV/publicselection. Global norm uses float64 diagnostic reduction, not bitwise torchglobalclip parity.'),f,indent=2)
 print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
