"""Frozen payload-only vs full-message intervention; run through run_safe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import horizon_token_language_engine as lab

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def score(model, data, args, mode):
    state = None
    generator = torch.Generator().manual_seed(args.seed + 100000)
    total, targets = 0., 0
    for begin in range(data.shape[1]-1):
        if state is not None and mode != 'intact':
            state = dict(state)
            fields = ('ctx_vals',) if mode == 'payload' else ('ctx_vals','ctx_arr','has_ctx')
            for field in fields:
                state[field] = torch.zeros_like(state[field])
        x,state,_ = model(data[:,begin:begin+1],state,generator,args,
                          deterministic=args.deterministic_eval)
        loss = model.readout.nll(x,data[:,begin+1:begin+2])
        total += float(loss.double().sum())
        targets += loss.numel()
    return total/targets,generator.get_state(),targets


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tags',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);rows=[]
    for tag in a.tags:
        path=ROOT/f'experiments/results/token_language/{tag}.json'
        result=json.loads(path.read_text())
        selected=json.loads(path.with_suffix('.selection.json').read_text())
        assert result['status']==selected['status']=='completed'
        for name,digest in result['identity']['source_sha256'].items():
            assert lab.sha(ROOT/name)==digest,name
        args=SimpleNamespace(**result['args'])
        train,dev=lab.load_tokens(args.train_file),lab.load_tokens(args.dev_file)
        counts=lab.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
        torch.manual_seed(args.seed)
        model=lab.Model(args,torch.argsort(counts,descending=True,stable=True),counts)
        checkpoint=Path(selected['selected']['checkpoint'])
        model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
        data=lab.interval_tensor(dev,args.dev_offset,args.dev_tokens,args.eval_lanes)
        scores={};rng=None
        for mode in ('intact','payload','full_message'):
            value,gen,targets=score(model,data,args,mode)
            assert targets==data.shape[0]*(data.shape[1]-1)
            if rng is None:rng=gen
            else:assert torch.equal(rng,gen)
            scores[mode]=value
        assert math.isclose(scores['intact'],selected['selected']['dev_nll'],abs_tol=2e-6)
        row=dict(tag=tag,seed=args.seed,selected=selected['selected'],scores=scores,
                 deltas={k:v-scores['intact'] for k,v in scores.items() if k!='intact'},
                 scored_targets=targets,matched_rng=True,partition_parity=True,
                 checkpoint_sha256=lab.sha(checkpoint))
        rows.append(row);print(json.dumps(row),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='completed',rows=rows,
        producer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Frozen selected checkpoints, development only. Payload intervention zeros ctx_vals while preserving ctx_arr and has_ctx before each event; full-message intervention zeros all three. Payload intervention preserves the normalization branch and incoming arrival metadata at that event, but future routes/timing/states can change. Matched route RNG and intact chunk-partition/selected-score parity. Neither intervention is a retrained architecture or a causal decomposition of total gains.'),indent=2)+'\n')


if __name__=='__main__':main()
