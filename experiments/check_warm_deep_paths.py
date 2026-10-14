"""R1 populated-state content cuts, preserving exact factual warm prefixes."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import allocated_depth_token_language as driver
from public_token_scoring import windows,reference_forward,score as original_score


@driver.torch.no_grad()
def score(model,tokens,args,start=25165824,targets=8192,length=1024,lanes=8,warm=256):
    model.eval();forward=reference_forward(model,driver)
    gen=driver.torch.Generator().manual_seed(args.seed+100000)
    total=0.;prefix=0.;count=0;prefix_count=0
    for inputs,labels,_ in windows(tokens,start,targets,length,lanes):
        state=None
        for pos in range(0,length,args.chunk):
            x,state,_=forward(inputs[:,pos:pos+args.chunk],state,gen,args,deterministic=args.deterministic_eval)
            losses=model.readout.nll(x,labels[:,pos:pos+args.chunk]).reshape(inputs.shape[0],-1).double()
            total+=float(losses.sum());count+=losses.numel()
            take=max(0,min(losses.shape[1],warm-pos))
            prefix+=float(losses[:,:take].sum());prefix_count+=inputs.shape[0]*take
    assert count==targets
    return dict(nll=total/count,loss_sum=total,targets=count,
                prefix_loss_sum=prefix,prefix_targets=prefix_count,
                tail_loss_sum=total-prefix,tail_targets=count-prefix_count,
                tail_nll=(total-prefix)/(count-prefix_count))


class WarmContentCut:
    def __init__(self,warm,length):
        if not 0<warm<length:raise ValueError('Nonempty factual prefix and cut suffix required')
        self.warm,self.length,self.calls=warm,length,0

    def __call__(self,module,inputs,output):
        position=self.calls%self.length;self.calls+=1
        return output if position<self.warm else driver.torch.zeros_like(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();driver.torch.set_num_threads(1);began=time.monotonic()
    base=ROOT/f'experiments/results/token_language/{a.run}'
    packet=json.loads(base.with_suffix('.json').read_text());assert packet['status']=='completed'
    for name,digest in packet['source_sha256'].items():assert driver.sha(ROOT/name)==digest,name
    args=SimpleNamespace(**packet['args']);assert args.depth>=8
    tr=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file)
    counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts()
    order=driver.torch.argsort(counts,descending=True,stable=True)
    driver.torch.manual_seed(args.seed);model=driver.Model(args,order,counts)
    checkpoint=driver.torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);del checkpoint
    factual=score(model,dev,args)
    reference=original_score(model,dev,args,driver,25165824,8192,1024,8)
    assert factual['loss_sum']==reference['loss_sum']
    rows=[]
    for layer,module in enumerate(model.core.channel_mix):
        cut=WarmContentCut(256,1024);handle=module.register_forward_hook(cut)
        try:intervention=score(model,dev,args)
        finally:handle.remove()
        assert cut.calls==1024,'Unexpected native stage invocation count'
        assert intervention['prefix_loss_sum']==factual['prefix_loss_sum'],'Cut changed earlier predictions'
        rows.append(dict(layer=layer,full_nll=intervention['nll'],tail_nll=intervention['tail_nll'],
                         tail_nll_increase=intervention['tail_nll']-factual['tail_nll'],prefix_parity=True))
    assert score(model,dev,args)==factual,'Hooks failed to restore factual scoring'
    result=dict(status='completed',battle='R1',run=a.run,factual=factual,
        warm_prefix_per_window=256,history=1024,stage_cuts=rows,
        diagnostic_scored_targets=8192*(args.depth+3),
        input_sha256={str(path.relative_to(ROOT)):driver.sha(path) for path in (base.with_suffix('.json'),base.with_suffix('.pt'))},
        source_sha256={str(path.relative_to(ROOT)):driver.sha(path) for path in (Path(__file__),ROOT/'experiments/public_token_scoring.py')},
        diagnostic_wall_s=time.monotonic()-began,
        scope='Single-seed final native checkpoint on reusedDEV. Retains256 factual inputs and populated banks before cutting one stage incoming projection output for768 targets. Clock parameters and memory retained; realized routes/times can change. Predictive dependence/retention under intervention, not trained shallow-model quality or added-depth information.')
    path=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
