"""Frozen state interventions; run only through the guarded one-job queue."""
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
def score(model,data,args,mode,chunk):
    state=None;gen=torch.Generator().manual_seed(args.seed+100000)
    total=0.;targets=0
    for start in range(0,data.shape[1]-1,chunk):
        n=min(chunk,data.shape[1]-1-start)
        if state is not None:
            if mode in ('no_memory','no_history'):
                state=dict(state)
                for key in ('mem','arr','seen'):
                    state[key]=[torch.zeros_like(v) for v in state[key]]
            if mode in ('no_message','no_history'):
                state=dict(state)
                for key in ('ctx_vals','ctx_arr','has_ctx'):
                    state[key]=torch.zeros_like(state[key])
        x,state,_=model(data[:,start:start+n],state,gen,args,deterministic=args.deterministic_eval)
        loss=model.readout.nll(x,data[:,start+1:start+n+1])
        total+=float(loss.sum());targets+=loss.numel()
    return total/targets,gen.get_state(),targets

torch.set_num_threads(1);rows=[]
folder=ROOT/'experiments/results/token_language'
for seed in (6,7):
    for window in (16,64):
        suffix='' if seed==6 else '_s7'
        tag=f'curie_horizon_2k_batch64_credit{window}{suffix}_20261005_v1'
        result=json.loads((folder/(tag+'.json')).read_text())
        selected=json.loads((folder/(tag+'.selection.json')).read_text())['selected']
        a=SimpleNamespace(**result['args'])
        train=lab.load_tokens(a.train_file);dev=lab.load_tokens(a.dev_file)
        counts=lab.LaneTokens(train,0,a.train_tokens,a.lanes).counts()
        torch.manual_seed(seed)
        model=lab.Model(a,torch.argsort(counts,descending=True,stable=True),counts)
        checkpoint=Path(selected['checkpoint'])
        model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
        data=lab.interval_tensor(dev,a.dev_offset,a.dev_tokens,a.eval_lanes)
        original,original_rng,targets=score(model,data,a,'intact',a.chunk)
        losses={};rngs={}
        for mode in ('intact','no_memory','no_message','no_history'):
            losses[mode],rngs[mode],count=score(model,data,a,mode,1)
            assert count==targets and torch.equal(rngs[mode],original_rng),(tag,mode,'RNG mismatch')
        assert math.isclose(original,selected['dev_nll'],abs_tol=2e-6)
        assert math.isclose(original,losses['intact'],abs_tol=2e-6),(tag,original,losses)
        rows.append(dict(tag=tag,seed=seed,credit_window=window,targets=targets,
            losses=losses,history_gains={mode:losses[mode]-losses['intact'] for mode in ('no_memory','no_message','no_history')},
            matched_rng=True,partition_parity=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        print(json.dumps(rows[-1]),flush=True)
out=ROOT/'experiments/results/diagnostics/curie_token_persistent_utility_audit_20261005_v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(dict(status='completed',rows=rows,
    scope='Frozen selected development checkpoints. Before every token, erase addressed memory and its arrival/seen markers, previous-event message, or both; retain global position and matched route RNG. Interventions are out of training distribution, not retrained ablations. No fitting or public validation.'),indent=2)+'\n')
