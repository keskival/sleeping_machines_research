"""R1 public scoring numerical contract on synthetic and development tokens only."""
import argparse,json,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import window_replay_token_language as driver
from public_token_scoring import windows,score,reference_forward

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
 toy=np.arange(131,dtype=np.uint16);covered=[]
 for x,y,begin in windows(toy,2,128,16,3):
  assert torch.equal(y,x+1);covered.extend(y.flatten().tolist())
 assert covered==list(range(3,131))
 base=ROOT/'experiments/results/token_language/aws_integrated_window_replay_512k_20261012_s6_v1';r=json.loads(base.with_suffix('.json').read_text());args=SimpleNamespace(**r['args'])
 train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file);counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
 torch.manual_seed(args.seed);model=driver.Model(args,order,counts);checkpoint=base.with_suffix('.pt');digest=driver.sha(checkpoint)
 acquired=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(acquired['model']);del acquired
 result=score(model,dev,args,driver,25165824,256,128,2)
 # Independent one-pass reference: same lane/window order and same route RNG,
 # one full forward versus chunked forward; targets include window-final token.
 model.eval();gen=torch.Generator().manual_seed(args.seed+100000);forward=reference_forward(model,driver)
 with torch.no_grad():
  x,y,_=next(windows(dev,25165824,256,128,2));features,_,_=forward(x,None,gen,args,deterministic=args.deterministic_eval)
  direct=float(model.readout.nll(features,y).double().sum())/256
 assert abs(direct-result['nll'])<1e-6
 assert driver.sha(checkpoint)==digest
 sources=dict(r['source_sha256'])
 for source in ('experiments/public_token_scoring.py',str(Path(__file__).relative_to(ROOT))):sources[source]=driver.sha(ROOT/source)
 for source,sha in sources.items():assert driver.sha(ROOT/source)==sha,source
 d=dict(status='completed',battle='R1',gate='PASS',synthetic_target_coverage=True,development_score=result,full_vs_chunked_nll_error=abs(direct-result['nll']),source_sha256=sources,input_sha256={str(base.with_suffix('.json').relative_to(ROOT)):driver.sha(base.with_suffix('.json')),str(checkpoint.relative_to(ROOT)):digest},scope='Synthetic indices and256 existing development targets only. Public validation untouched. Contract verifies shifted targets, window-final target coverage, reset only at window boundary, EOS ordinary token, normalized native readout and full/chunk score agreement. Public scorer defaults1024 context; small contract128 context is not a public benchmark score.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
