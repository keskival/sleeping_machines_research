"""Real driver interruption/continuation contract inside one guarded diagnostic job."""
import json,sys,gc
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import integrated_token_language_lab as lab
base=['--readout-init','frequency','--input-init','balanced','--steps','6','--eval-every','3',
      '--train-tokens','512','--dev-tokens','128','--payload','8','--heads','1','--depth','2',
      '--pool','2','--lanes','2','--eval-lanes','2','--chunk','4','--future-every','1']
folder=ROOT/'experiments/results/token_language'
for tag,extra in [('aws_integrated_resume_full_20261005T181000Z_v1',[]),
                  ('aws_integrated_resume_partial_20261005T181000Z_v1',['--stop-after-step','3']),
                  ('aws_integrated_resume_restored_20261005T181000Z_v1',['--resume',str(folder/'aws_integrated_resume_partial_20261005T181000Z_v1.pt')])]:
    sys.argv=['integrated_token_language_lab.py','--tag',tag,*base,*extra]
    lab.main();gc.collect()
a=torch.load(folder/'aws_integrated_resume_full_20261005T181000Z_v1.pt',weights_only=False)
b=torch.load(folder/'aws_integrated_resume_restored_20261005T181000Z_v1.pt',weights_only=False)
def equal(x,y):
    if isinstance(x,torch.Tensor):assert torch.equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:equal(x[k],y[k])
    elif isinstance(x,(list,tuple)):
        assert len(x)==len(y)
        for xx,yy in zip(x,y):equal(xx,yy)
    else:assert x==y
for key in ('model','optimizer','state','generator','alternative_generator','local_generator','torch_rng','cursor','step','total_presentations','writes'):
    equal(a[key],b[key])
assert a['total_presentations']==b['total_presentations']==48
for ar,br in zip(a['curve'],b['curve']):
    for key in ('step','train_nll','dev_nll'):equal(ar[key],br[key])
result=dict(status='completed',contract='Actual token lab uninterrupted vs optimizer-boundary interruption/resume',
            exact_fields=['model','optimizer','numeric state','factual RNG','future-alternative RNG','local-value-alternative RNG','global RNG','cursor','quality curve','48 cumulative presentations'],
            scope='One guarded diagnostic job; three tiny sequential paths; includes sparse local teacher and actual alternative-write teacher')
p=ROOT/'experiments/results/diagnostics/aws_integrated_token_resume_contract_20261005T181000Z_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
