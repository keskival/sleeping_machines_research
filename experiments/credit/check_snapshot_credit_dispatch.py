"""R1 actual D8 factual and alternative route producer update parity."""
import argparse,copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import unrestricted_receiver_token_language as driver
from sleeping_machines.learning.receiver_credit_banks import ReceiverCreditBanks
from sleeping_machines.learning.snapshot_credit_dispatch import SnapshotCreditDispatch
from sleeping_machines.paired_route_credit import paired_route_credit

def losses(model,a):
 ids=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[2,3,4,5]])
 g=torch.Generator().manual_seed(7);initial=g.get_state();x,state,pis=model(ids,None,g,a,deterministic=True)
 nll=model.readout.nll(x,labels).reshape(1,4)
 pi=pis[0][1][:,0,:].double();winner=state['race_winners'][0,0,:,0];alt=(winner+1)%a.pool
 model.force_site=(0,0,0,alt);model.shadow_mode=True;model.suppress_site=(0,0,0)
 with torch.no_grad():
  replay=torch.Generator();replay.set_state(initial);shadow,_,_=model(ids,None,replay,a,deterministic=True)
  delta=(model.readout.nll(shadow,labels).reshape(1,4)-nll.detach()).mean(1)
 model.force_site=model.suppress_site=None;model.shadow_mode=False
 teacher=paired_route_credit(pi,alt,torch.full_like(delta,1/(a.pool-1)),delta)
 return [nll[:,:2].sum()/4,nll[:,2:].sum()/4,teacher]

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;torch.set_num_threads(1)
 a=SimpleNamespace(**json.loads((ROOT/'experiments/results/token_language/aws_native_deep_d8_8m_20261013_s6_v1.json').read_text())['args']);a.payload=8;a.credit_window=4;a.eos_reset=False
 counts=torch.ones(50257,dtype=torch.long);order=torch.arange(50257);torch.manual_seed(6);live=driver.Model(a,order,counts);reference=copy.deepcopy(live)
 banks=ReceiverCreditBanks(live,lr=.003,weight_decay=.01,local_clip=1.);ref=ReceiverCreditBanks(reference,lr=.003,weight_decay=.01,local_clip=1.)
 receipts=[];max_error=0.;moment_error=0.;started=time.monotonic()
 for update in range(3):
  dispatch=SnapshotCreditDispatch(live,banks,f'u{update}',('factual_early','factual_late','alternative_route'))
  snapshot_before={n:p.detach().clone() for n,p in dispatch.actor.named_parameters()}
  terms=losses(dispatch.actor,a)
  reference.zero_grad(set_to_none=True);baseline=losses(reference,a);sum(baseline).backward()
  for packet in ref.packets(f'u{update}'):ref.apply(packet)
  for i,name in enumerate(dispatch.producers):dispatch.backward(name,terms[i],retain_graph=i<2)
  receipts.append(dispatch.finish());assert receipts[-1]['applied_during_backward']>0
  for n,p in dispatch.actor.named_parameters():assert torch.equal(p,snapshot_before[n]),'Actor snapshot changed during early updates'
  for n,p in live.named_parameters():max_error=max(max_error,float((p-reference.state_dict()[n]).abs().max()))
  for bank,opt in banks.local.optimizers.items():
   other=ref.local.optimizers[bank]
   for state,expected in zip(opt.state.values(),other.state.values()):
    for k,v in state.items():
     if isinstance(v,torch.Tensor):moment_error=max(moment_error,float((v-expected[k]).abs().max()))
  assert banks.local.versions==ref.local.versions
  live.zero_grad(set_to_none=True);banks.local.receipts.clear();ref.local.receipts.clear()
 assert max_error<1e-6 and moment_error<1e-7,(max_error,moment_error)
 sources={n:driver.sha(ROOT/n) for n in ['experiments/credit/check_snapshot_credit_dispatch.py','experiments/unrestricted_receiver_token_language.py','sleeping_machines/learning/snapshot_credit_dispatch.py','sleeping_machines/learning/receiver_credit_banks.py','sleeping_machines/learning/local_credit_banks.py','sleeping_machines/learning/bank_credit_inbox.py','sleeping_machines/paired_route_credit.py']}
 out=ROOT/f'experiments/results/credit/{tag}.json'
 with out.open('x') as f:json.dump(dict(status='completed',battle='R1',gate='PASS',parameter_error=max_error,moment_error=moment_error,receipts=receipts,wall_s=time.monotonic()-started,source_sha256=sources,scope='Three D8 synthetic4token native updates with actual forced-route replay, two factual producers and route surrogate; fixed alternative cyclic proposal is a diagnostic, not production policy. Local clip/AdamW; immutable snapshot. Floating reduction tolerance, no corpus quality or faster execution claim.'),f,indent=2)
if __name__=='__main__':main()
