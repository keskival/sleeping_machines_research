"""R1 frozen-weight persistent-history interventions on existing development panels."""
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path
from types import SimpleNamespace, FunctionType
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import window_replay_token_language as driver
from sleeping_machines.sparse_counterfactual_episodes import initial_state


def history_forward(model, span):
    original=driver.Model.forward
    namespace=dict(original.__globals__);real=namespace['token_features'];calls=[0]
    def intervened(core, ids, state, generator, **kwargs):
        # Every original call consumes one credit window (16 tokens). Retain
        # stream position; remove stored content, arrivals, seen and context.
        if span and ids.shape[1]>span:
            parts=[];probabilities=[]
            for begin in range(0,ids.shape[1],span):
                x,state,pis=intervened(core,ids[:,begin:begin+span],state,generator,**kwargs)
                parts.append(x);probabilities.extend(pis)
            return torch.cat(parts,1),state,probabilities
        if span and calls[0] and calls[0]%span==0:
            old=state;state=initial_state(core,ids.shape[0])
            if old is not None:state['position']=old['position'].clone()
        calls[0]+=ids.shape[1]
        return real(core,ids,state,generator,**kwargs)
    namespace['token_features']=intervened
    fn=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    fn.__kwdefaults__=original.__kwdefaults__
    return fn.__get__(model,type(model))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    reference_path=ROOT/'experiments/results/diagnostics/aws_fresh_scaling_transfer_20261013_v1.json'
    reference=json.loads(reference_path.read_text());assert reference['status']=='completed'
    sources={};inputs={str(reference_path.relative_to(ROOT)):driver.sha(reference_path)};rows=[]
    for seed in (6,7,8):
        for size in ('512k','8m'):
            tag=(f'aws_integrated_window_replay_512k_20261012_s{seed}_v1' if size=='512k' else
                 f'aws_integrated_fresh_scaling_8m_pool4_{"20261012" if seed==6 else "20261013"}_s{seed}_v1')
            base=ROOT/f'experiments/results/token_language/{tag}';r=json.loads(base.with_suffix('.json').read_text())
            assert r['status']=='completed'
            for name,digest in r['source_sha256'].items():
                assert driver.sha(ROOT/name)==digest,name;sources[name]=digest
            inputs[str(base.with_suffix('.json').relative_to(ROOT))]=driver.sha(base.with_suffix('.json'))
            args=SimpleNamespace(**r['args']);assert args.credit_window==16 and args.chunk==64 and args.state_mode=='carry'
            train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file)
            counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
            torch.manual_seed(seed);model=driver.Model(args,order,counts)
            checkpoint=base.with_suffix('.pt');inputs[str(checkpoint.relative_to(ROOT))]=driver.sha(checkpoint)
            acquired=torch.load(checkpoint,map_location='cpu',weights_only=False);assert acquired['step']==args.steps
            model.load_state_dict(acquired['model']);del acquired
            previous=next(x for x in reference['rows'] if x['seed']==seed and x['acquisition']==size)
            scores={};started=time.monotonic()
            # Same first four existing panels: diagnostic re-use, no selection.
            for span in (0,1,4):
                panels=[]
                for i in range(4):
                    model.forward=history_forward(model,span)
                    data=driver.interval_tensor(dev,25165824+i*2048,2048,args.eval_lanes)
                    panels.append(driver.evaluate(model,data,args))
                scores[str(span)]=panels
            assert scores['0']==previous['panel_nll'][:4], 'Factual route/readout partition changed'
            assert driver.sha(checkpoint)==inputs[str(checkpoint.relative_to(ROOT))]
            rows.append(dict(seed=seed,acquisition=size,scored_targets=8160,panel_nll=scores,
                nll={k:statistics.mean(v) for k,v in scores.items()},wall_s=time.monotonic()-started))
            print(json.dumps(rows[-1]),flush=True)
    paired=[]
    for seed in (6,7,8):
        before,after=[next(x for x in rows if x['seed']==seed and x['acquisition']==size) for size in ('512k','8m')]
        paired.append(dict(seed=seed,acquisition_gain={k:before['nll'][k]-after['nll'][k] for k in ('0','1','4')},
            history_value_before={k:before['nll'][k]-before['nll']['0'] for k in ('1','4')},
            history_value_after={k:after['nll'][k]-after['nll']['0'] for k in ('1','4')}))
    sources[str(Path(__file__).relative_to(ROOT))]=driver.sha(Path(__file__))
    source='sleeping_machines/sparse_counterfactual_episodes.py';sources[source]=driver.sha(ROOT/source)
    d=dict(status='completed',battle='R1',rows=rows,paired=paired,source_sha256=sources,input_sha256=inputs,
        scope='Three seeds, frozen 512k/8M weights, same first four existing transfer DEV panels (8160 targets). Reset complete stored receiver/content/arrival/context state every1 or4 observed tokens, retain stream position and same explicit route RNG. Full-history factual panels must reproduce exactly. Reset is a distribution intervention, not an independently trained short-history control; signed loss differences measure reliance on this history under the intervention, not isolated memory capacity or a scaling exponent. Public validation sealed; no fitting or selection.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')

if __name__=='__main__':main()
