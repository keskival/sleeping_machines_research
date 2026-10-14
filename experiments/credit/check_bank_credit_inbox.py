"""R1 independent credit readiness, canonical sum and stale/duplicate rejection."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sleeping_machines.learning.local_credit_banks import LocalCreditBanks
from sleeping_machines.learning.bank_credit_inbox import BankCreditInbox,CreditFragment

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic()
 out=ROOT/f'experiments/results/credit/{a.tag}.json'
 if out.exists():raise FileExistsError(out)
 p=torch.nn.Parameter(torch.zeros(2));q=torch.nn.Parameter(torch.ones(2));banks=LocalCreditBanks([('a',p),('b',q)],local_clip=1.,weight_decay=.01)
 box=BankCreditInbox(banks);box.issue('a',0,'u0',('x','y'));box.issue('b',0,'u0',('z',))
 first=CreditFragment('a',0,'y','u0',torch.tensor([1.,-2.]));assert not box.receive(first)
 rejected=[]
 for label,fragment in [('duplicate',first),('version',CreditFragment('a',1,'x','u0',torch.ones(2))),('event',CreditFragment('a',0,'x','other',torch.ones(2)))]:
  try:box.receive(fragment)
  except ValueError:rejected.append(label)
  else:raise AssertionError(label)
 try:box.take('a')
 except ValueError:rejected.append('incomplete')
 else:raise AssertionError('incomplete')
 assert box.receive(CreditFragment('b',0,'z','u0',torch.ones(2)));before=p.detach().clone();banks.apply(box.take('b'))
 assert banks.versions=={'a':0,'b':1} and torch.equal(p,before),'Unready bank must not hold back ready bank'
 assert box.receive(CreditFragment('a',0,'x','u0',torch.tensor([2.,3.])))
 packet=box.take('a');assert torch.equal(packet.gradient,torch.tensor([3.,1.]));banks.apply(packet)
 assert box.storage()['retained_gradient_bytes']==0
 sources=['experiments/credit/check_bank_credit_inbox.py','sleeping_machines/learning/bank_credit_inbox.py','sleeping_machines/learning/local_credit_banks.py']
 result=dict(status='completed',battle='R1',tag=a.tag,gate='PASS',independent_bank_readiness=True,canonical_sum_error=0.,rejections=rejected,storage=box.storage(),wall_s=time.monotonic()-start,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Declared producer completion sets and bank-local clippedAdamW. No actual native event-gradient production, concurrent actor reads, physical hardware or corpus-quality claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
