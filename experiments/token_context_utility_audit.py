"""Frozen dev context utility against the same learned constant-feature readout."""
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
import horizon_token_language_engine as lab


@torch.no_grad()
def features(model,data,args):
    state=None;g=torch.Generator().manual_seed(args.seed+100000)
    xs=[];ys=[]
    for start in range(0,data.shape[1]-1,args.chunk):
        n=min(args.chunk,data.shape[1]-1-start)
        x,state,_=model(data[:,start:start+n],state,g,args,deterministic=args.deterministic_eval)
        xs.append(x.reshape(-1,x.shape[-1]));ys.append(data[:,start+1:start+n+1].reshape(-1))
    return torch.cat(xs),torch.cat(ys)


torch.set_num_threads(1);rows=[]
for seed in (6,7):
    for window in (16,64):
        suffix='' if seed==6 else '_s7'
        tag=f'curie_horizon_2k_batch64_credit{window}{suffix}_20261005_v1'
        folder=ROOT/'experiments/results/token_language'
        result=json.loads((folder/(tag+'.json')).read_text())
        selected=json.loads((folder/(tag+'.selection.json')).read_text())['selected']
        a=SimpleNamespace(**result['args']);train=lab.load_tokens(a.train_file);dev=lab.load_tokens(a.dev_file)
        counts=lab.LaneTokens(train,0,a.train_tokens,a.lanes).counts()
        torch.manual_seed(seed)
        model=lab.Model(a,torch.argsort(counts,descending=True,stable=True),counts)
        checkpoint=Path(selected['checkpoint'])
        model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
        train_data=lab.interval_tensor(train,0,a.train_tokens,a.lanes)
        dev_data=lab.interval_tensor(dev,a.dev_offset,a.dev_tokens,a.eval_lanes)
        with torch.no_grad():
            train_x,_=features(model,train_data,a)
            x,y=features(model,dev_data,a)
            losses={}
            for name,values in [('contextual',x),('zero_features',torch.zeros_like(x)),
                                ('train_mean_features',train_x.mean(0).expand_as(x))]:
                losses[name]=float(model.readout.nll(values,y).mean())
        assert math.isclose(losses['contextual'],selected['dev_nll'],abs_tol=2e-6)
        rows.append(dict(tag=tag,seed=seed,credit_window=window,targets=y.numel(),
            train_feature_observations=train_x.shape[0],losses=losses,
            gain_over_training_mean=losses['train_mean_features']-losses['contextual'],
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
out=ROOT/'experiments/results/diagnostics/curie_token_context_utility_audit_20261005_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(status='completed',rows=rows,
    scope='Frozen development intervention on selected seed6/7 checkpoints; same readout, training-only feature mean. Constant features remove token/state dependence but need not be an optimally refitted unigram control. No fitting/public validation. Actual dev count1016 from1024admitted tokens and8lane starts.'),indent=2)+'\n')
print(json.dumps(rows),flush=True)
