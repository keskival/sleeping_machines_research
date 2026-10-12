"""R1 fixed-target distinct-experience decision with contextual utility."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import torch
import horizon_token_language_engine as lab


@torch.no_grad()
def features(model,data,args):
    state=None; gen=torch.Generator().manual_seed(args.seed+100000); xs=[]; ys=[]
    for begin in range(0,data.shape[1]-1,args.chunk):
        n=min(args.chunk,data.shape[1]-1-begin)
        x,state,_=model(data[:,begin:begin+n],state,gen,args,deterministic=args.deterministic_eval)
        xs.append(x.reshape(-1,x.shape[-1]));ys.append(data[:,begin+1:begin+n+1].reshape(-1))
    return torch.cat(xs),torch.cat(ys)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    a=p.parse_args();torch.set_num_threads(1);began=time.perf_counter();folder=lab.ROOT/'experiments/results/token_language'
    records=[];rows=[];inputs={}
    for label,tag in [('repeated',a.reference),('distinct',a.candidate)]:
        path=folder/f'{tag}.json';blob=path.read_bytes();d=json.loads(blob);assert d['status']=='completed'
        inputs[str(path.relative_to(lab.ROOT))]=hashlib.sha256(blob).hexdigest()
        for name,digest in d['source_sha256'].items():assert lab.sha(lab.ROOT/name)==digest,name
        args=SimpleNamespace(**d['args']);torch.manual_seed(args.seed)
        train=lab.load_tokens(args.train_file);dev=lab.load_tokens(args.dev_file)
        assert lab.sha(args.train_file)==d['identity']['train_sha256']
        assert lab.sha(args.dev_file)==d['identity']['dev_sha256']
        counts=lab.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
        order=torch.argsort(counts,descending=True,stable=True)
        model=lab.Model(args,order,counts)
        selected=min(d['curve'],key=lambda r:r['dev_nll'])
        if selected['step']:
            checkpoint=Path(d['best_checkpoint']);model.load_state_dict(torch.load(checkpoint,weights_only=True))
            inputs[str(checkpoint.relative_to(lab.ROOT))]=lab.sha(checkpoint)
        model.eval()
        x,y=features(model,lab.interval_tensor(dev,args.dev_offset,args.dev_tokens,args.eval_lanes),args)
        with torch.no_grad():actual=float(model.readout.nll(x,y).mean())
        assert abs(actual-selected['dev_nll'])<3e-6,(actual,selected['dev_nll'])
        # Same causal TRAIN probe for both models; no DEV targets fit this comparator.
        probe,_=features(model,lab.interval_tensor(train,0,8192,args.eval_lanes),args)
        mean=probe.double().mean(0).to(x.dtype)
        with torch.no_grad():constant=float(model.readout.nll(mean.expand_as(x),y).mean())
        rows.append(dict(arm=label,unique_tokens=args.train_tokens,fitting_targets=d['presentations_total'],
            updates=args.steps,initial_dev_nll=d['curve'][0]['dev_nll'],selected_dev_nll=actual,
            selected_step=selected['step'],constant_train_feature_nll=constant,context_gain=constant-actual,
            learning_gain=d['curve'][0]['dev_nll']-actual,training_wall_s=d['train_wall_s_total'],
            engine_wall_s=d['wall_s'],process_wall_s=d['profile']['process_wall_s']))
        records.append(d)
    reference,candidate=records
    assert {k:v for k,v in reference['args'].items() if k not in ('tag','train_tokens')}=={k:v for k,v in candidate['args'].items() if k not in ('tag','train_tokens')}
    assert reference['identity']==candidate['identity']
    assert reference['presentations_total']==candidate['presentations_total']==131056
    assert reference['host']==candidate['host']
    gain=rows[0]['selected_dev_nll']-rows[1]['selected_dev_nll']
    sources={'experiments/analyze_unique_token_stream.py':lab.sha(__file__),**reference['source_sha256']}
    result=dict(status='completed',battle='R1',rows=rows,quality_gain=gain,
        initial_prior_gain=rows[0]['initial_dev_nll']-rows[1]['initial_dev_nll'],
        learning_gain_change=rows[1]['learning_gain']-rows[0]['learning_gain'],
        context_gain_change=rows[1]['context_gain']-rows[0]['context_gain'],
        development_gate_passed=gain>0 and rows[1]['context_gain']>0,
        wall_s=time.perf_counter()-began,input_sha256=inputs,source_sha256=sources,
        scope='Single-seed private DEV; same CPU model/updates/131056 fitting targets. Data diversity, lane partition and corpus-wrap resets differ intentionally. TRAIN-only frequency prior uses each admitted prefix and is included in process cost. Initial-prior and learned gains separated. Context comparator is the same fixed readout on the mean causal feature from an8192-token TRAIN probe, not a refitted unigram or isolated memory ablation. No iso-data, training-FLOP, Transformer or nonstationary-stream claim.')
    out=folder/f'{a.tag}.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
