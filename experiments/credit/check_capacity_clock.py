"""R1 capacity-clock law, common clock credit and exact cached replay checks."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import check_window_replay as four
import check_capacity_window_replay as eight
import capacity_clock_token_language as candidate
import window_replay_token_language as incumbent
from sleeping_machines import capacity_clock_episodes as clocks


def clock_law():
    args=SimpleNamespace(seed=6,payload=4,depth=2,heads=2,pool=8,input_init='balanced',
        tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,
        credit_window=8,state_mode='carry',route_credit='sampled',future_site='uniform',
        free_bias=0.,temperature=1.,compiled=False)
    torch.manual_seed(6)
    model=incumbent.Model(args,torch.arange(50257),torch.ones(50257,dtype=torch.long))
    namespace=clocks.original_features.__globals__;original=namespace['sparse_counterfactual_layer'];calls=[]
    def capture(*values,**options):
        calls.append((values,options));return original(*values,**options)
    namespace['sparse_counterfactual_layer']=capture
    try:model(torch.ones((2,1),dtype=torch.long),None,torch.Generator().manual_seed(8),args)
    finally:namespace['sparse_counterfactual_layer']=original
    values,options=calls[0]
    values=[v.detach().double().clone() if isinstance(v,torch.Tensor) and v.is_floating_point()
        else v.detach().clone() if isinstance(v,torch.Tensor) else v for v in values]
    values[14].requires_grad_()
    base=clocks.original_layer(*values,**options)
    scaled=clocks.calibrated_layer(*values,reference_pool=4,**options)
    for i in (2,3,4,5,6,8,9):assert torch.equal(base[i],scaled[i]),i
    noise=values[7];bias=values[14].reshape(2,8)
    time=2*(noise/bias.exp()).min(-1).values
    expected=values[1][:,None]+.001+.010*time/(1+time)
    error=float((scaled[7]-expected).abs().max())
    assert error<1e-14
    gradient=torch.autograd.grad(scaled[7].sum(),values[14])[0].sum()
    analytic=(-.010*time/(1+time).square()).sum()*values[0].shape[0]
    assert float((gradient-analytic).abs())<1e-13
    losses=[]
    for sign in (-1,1):
        shifted=list(values);shifted[14]=values[14].detach()+sign*1e-6
        losses.append(clocks.calibrated_layer(*shifted,reference_pool=4,**options)[7].sum())
    finite_difference=(losses[1]-losses[0])/2e-6
    fd_error=float((gradient-finite_difference).abs())
    assert fd_error<1e-8
    return dict(pool=8,reference_pool=4,clock_scale=2,clock_law_error=error,
        common_clock_gradient=float(gradient),finite_difference_error=fd_error,
        same_single_layer_probabilities_winners_values_and_writes=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    law=clock_law();rows=[]
    four.candidate=candidate
    for depth in (2,4):
        for mode in ('carry','reset'):rows.append(dict(pool=4,**four.check(depth,mode)))
    # Full-prefix reference and cached teacher both use the declared normalized
    # clocks. Only the pool4 cases compare with the unmodified incumbent.
    eight.candidate=candidate
    eight.reference.token_features=functools.partial(clocks.token_features,clock_reference_pool=4)
    for depth in (2,4):
        for mode in ('carry','reset'):rows.append(dict(pool=8,**eight.check(depth,mode)))
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix=='.py':
                relative=path.relative_to(ROOT)
                if relative.parts[0] in ('experiments','sleeping_machines'):sources[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',rows=rows,clock_law=law,source_sha256=sources,
        scope='Numerical gate: pool4 unchanged execution; pool8 declared clock law, same one-layer route probabilities/winners/values/writes, float64 common-intensity derivative and finite difference. Full-prefix/cached calibrated replay matches states, utility, RNG, gradients and Adam parameters across depth2/4,carry/reset. Longer network trajectories change with clocks. No quality, efficient learning or transfer claim.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'clock_law':law,'rows':rows}),flush=True)


if __name__=='__main__':main()
