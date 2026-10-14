"""R1 independent bank clipping, manual AdamW rule and exact local resume."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sleeping_machines.learning.local_credit_banks import CreditPacket,LocalCreditBanks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 out=ROOT/f'experiments/results/credit/{a.tag}.json'
 if out.exists():raise FileExistsError(out)
 rows=[]
 for other_scale in (1.,1000.):
  p=torch.nn.Parameter(torch.tensor([.2,-.3]));q=torch.nn.Parameter(torch.tensor([.4,.1]));expected=torch.nn.Parameter(p.detach().clone())
  banks=LocalCreditBanks([('a',p),('b',q)],lr=.003,weight_decay=.01,local_clip=.5)
  opt=torch.optim.AdamW([expected],lr=.003,weight_decay=.01)
  for k in range(3):
   raw=torch.tensor([3.+k,-4.]);clipped=raw*min(1.,.5/(float(raw.norm())+1e-6))
   expected.grad=clipped;opt.step();opt.zero_grad()
   # Reverse delivery order on alternate updates; unrelated gradient magnitude varies.
   packets=[CreditPacket('a',k,raw,f'u{k}'),CreditPacket('b',k,torch.tensor([other_scale,-other_scale]),f'u{k}')]
   for packet in (packets if k%2 else reversed(packets)):banks.apply(packet)
   assert torch.equal(p,expected),'Local clip differs from declared manual AdamW rule'
   if k==1:
    saved={name:copy.deepcopy(o.state_dict()) for name,o in banks.optimizers.items()};versions=dict(banks.versions)
    banks=LocalCreditBanks([('a',p),('b',q)],lr=.003,weight_decay=.01,local_clip=.5)
    for name,o in banks.optimizers.items():o.load_state_dict(saved[name])
    banks.versions=versions
  rows.append(p.detach().tolist())
 assert rows[0]==rows[1],'Unrelated bank gradient changed local update'
 result=dict(status='completed',battle='R1',tag=a.tag,gate='PASS',manual_parameter_error=0.,cross_bank_dependency_error=0.,scope='Three updates, exact per-bank L2clip0.5/AdamW0.01decay; reversed independent delivery order, unrelated gradient scale1vs1000. Does not claim parity with global clipping or corpus quality.',wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['experiments/credit/check_local_clip_independence.py','sleeping_machines/learning/local_credit_banks.py']})
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
