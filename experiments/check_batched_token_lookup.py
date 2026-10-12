"""R1 integration gate: batched lookup preserves causal state and learning."""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import horizon_token_language_engine as engine
import batched_token_episodes as batched


def error(a, b):
    if isinstance(a, dict):
        return max(error(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return max((error(x, y) for x, y in zip(a, b)), default=0.)
    if torch.is_tensor(a):
        if a.dtype == torch.bool:
            return float(not torch.equal(a, b))
        return float((a-b).abs().max()) if a.numel() else 0.
    return float(a != b)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(6); began = time.perf_counter()
    args = SimpleNamespace(seed=6,payload=24,depth=2,heads=2,pool=4,input_init='balanced',
        tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
        credit_window=16,state_mode='carry',route_credit='sampled',future_site='first',
        free_bias=0.,temperature=1.,compiled=False)
    counts = torch.ones(50257, dtype=torch.long)
    left=engine.Model(args,torch.arange(50257),counts); right=copy.deepcopy(left)
    opts=[torch.optim.AdamW(m.parameters(),lr=.003) for m in (left,right)]
    gens=[torch.Generator().manual_seed(7) for _ in range(2)]
    states=[None,None]; rows=[]; native=engine.token_features
    # Repeated IDs exercise gradient collisions; EOS resets occur inside/across windows.
    tokens=torch.tensor([[3,3,9,50256,3,9,9,3]*4,[9,3,3,9,3,50256,9,3]*4])
    for update in range(3):
        outputs=[]; gradients=[]
        for i,m in enumerate((left,right)):
            engine.token_features=native if i==0 else batched.token_features
            opts[i].zero_grad(set_to_none=True)
            x,states[i],pi=m(tokens,states[i],gens[i],args)
            loss=m.readout.nll(x,tokens.roll(-1,1)).mean()
            loss.backward()
            gradients.append({n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None})
            outputs.append((x.detach(),states[i],pi,loss.detach()))
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opts[i].step()
            states[i]=engine.detach(states[i])
        row=dict(update=update,forward_state_error=error(outputs[0],outputs[1]),
                 gradient_error=error(gradients[0],gradients[1]),
                 parameter_error=error(left.state_dict(),right.state_dict()),
                 optimizer_error=error(opts[0].state_dict(),opts[1].state_dict()),
                 rng_equal=torch.equal(gens[0].get_state(),gens[1].get_state()) and
                 torch.equal(left.local_generator.get_state(),right.local_generator.get_state()))
        rows.append(row)
        if max(row[k] for k in ('forward_state_error','gradient_error','parameter_error','optimizer_error')) > 2e-5 or not row['rng_equal']:
            raise AssertionError(row)
    engine.token_features=native
    sources=['experiments/check_batched_token_lookup.py','experiments/batched_token_episodes.py',
             'experiments/horizon_token_language_engine.py','sleeping_machines/sparse_counterfactual_episodes.py']
    out=engine.ROOT/'experiments/results/token_language'/f'{a.tag}.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(dict(status='completed',battle='R1',rows=rows,wall_s=time.perf_counter()-began,
        source_sha256={s:engine.sha(engine.ROOT/s) for s in sources},
        scope='Three integrated AdamW updates; carried state, repeated tokens, EOS, two credit windows and sampled local alternative credit. Future suffix replay measured separately in whole-step profile.'),indent=2)+'\n')


if __name__=='__main__': main()
