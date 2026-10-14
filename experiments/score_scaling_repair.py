"""Same8192DEV/1024history scoring for contained native repair checkpoints."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import scaling_repair_continuation as repair
from public_token_scoring import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--run',required=True);a=p.parse_args();torch.set_num_threads(1);base=ROOT/f'experiments/results/token_language/{a.run}';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed';sources=dict(r['source_sha256'])
 for name,digest in sources.items():assert repair.driver.sha(ROOT/name)==digest,name
 args=SimpleNamespace(**r['args']);tr=repair.driver.load_tokens(args.train_file);dev=repair.driver.load_tokens(args.dev_file);counts=repair.driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();torch.manual_seed(args.seed);model=repair.build_model(args,torch.argsort(counts,descending=True,stable=True),counts);ck=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False);assert ck['step']==args.steps;model.load_state_dict(ck['model']);del ck
 original=repair.driver.evaluate(model,repair.driver.interval_tensor(dev,args.dev_offset,args.dev_tokens,args.eval_lanes),args);assert original==r['curve'][-1]['dev_nll']
 began=time.monotonic();scored=score(model,dev,args,repair.driver,25165824,8192,1024,8)
 for name in ['experiments/score_scaling_repair.py','experiments/public_token_scoring.py']:sources[name]=repair.driver.sha(ROOT/name)
 d=dict(status='completed',battle='R1',variant=r['variant'],scored=scored,parameters=r['parameters'],added_targets=r['added_targets'],added_training_s=r['added_training_s'],diagnostic_s=time.monotonic()-began,source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):repair.driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},scope='Finalweights/seed6/8192reusedDEV/1024history/noEOSreset, ownnative repair compared with sharedglobal continuation atsame acquisition; no publicvalidation selection. Decoder uses onlyfinalstage features, allcosts retained.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2)
 print(json.dumps(scored))
if __name__=='__main__':main()
