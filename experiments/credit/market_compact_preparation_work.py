"""Executed arithmetic of compact2048target native guidance preparation."""
import argparse,json,copy
import market_native_compact_acquisition as e
import market_complete_work_probe as probe
from market_complete_work_probe_v2 import LikelihoodAudit
from numpy_work_ledger import NumpyProxy
import market_clock_context_contract as clocks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    e.torch.set_num_threads(1);e.torch.set_default_dtype(e.torch.float64);probe.TorchAudit=LikelihoodAudit
    admitted=e.ROOT/'experiments/results/market/curie_b10_compact_acquisition_20261013_v1_fit.json'
    proof=json.loads(admitted.read_text());assert proof['status']=='completed'
    assert max(proof['deployment_parity'].values())<1e-12
    model,inputs=e.restored();t,m,path=e.load('2026-08-24',128+2048)
    inputs[str(path.relative_to(e.ROOT))]=e.sha(path);e.torch.manual_seed(19101)
    critic=e.NativeMarketCredit(264).double();critic.core.head=e.torch.nn.Linear(critic.core.total_payload,1).double()
    e.torch.nn.init.zeros_(critic.core.head.weight);e.torch.nn.init.zeros_(critic.core.head.bias)
    initial=(model,e.torch.optim.Adam(model.parameters(),lr=.0003),e.warm(model,t,m,128),critic,e.torch.optim.Adam(critic.parameters(),lr=.001),.001)
    rng=e.torch.get_rng_state().clone();previous_np=clocks.np
    try:
        with probe.instrumentation() as (ta,na):
            clocks.np=NumpyProxy(na)
            bundle,source=e.stream(initial,t,m,127,64,32,'online',True)
    finally:clocks.np=previous_np
    torch_work,numpy_work=ta.result(),na.result()
    # Direct ndarray frequency*elapsed is outside the proxy; count its exact
    # one multiply per unit per compact-source event explicitly.
    compact_events=(64-source['remote_windows'])*32
    frequency_products=sum(l.n for l in model.layers)*compact_events
    arithmetic=torch_work['arithmetic_flops']+numpy_work['arithmetic_flops']+frequency_products
    e.torch.set_rng_state(rng);reference,_=e.stream(initial,t,m,127,64,32,'online',True)
    parity={}
    for key,index in [('actor',0),('critic',3)]:
        parity[key]=max(float((a-b).abs().max()) for a,b in zip(bundle[index].parameters(),reference[index].parameters()))
    assert max(parity.values())<1e-12
    parent=e.ROOT/'experiments/results/market/curie_b10_actor_boundary_20261013_v1.json'
    old=json.loads(parent.read_text());before=old['guidance_acquisition_arithmetic_flops']
    files=['experiments/credit/market_compact_preparation_work.py','experiments/credit/market_native_compact_acquisition.py',
           'experiments/credit/market_clock_context_contract.py','experiments/credit/market_complete_work_probe.py',
           'experiments/credit/market_complete_work_probe_v2.py','experiments/credit/numpy_work_ledger.py']
    result=dict(status='completed',battle='B10',source=source,torch_work=torch_work,numpy_work=numpy_work,
        explicitly_counted_frequency_products=frequency_products,compact_events=compact_events,
        acquisition_arithmetic_flops=arithmetic,previous_acquisition_arithmetic_flops=before,
        saved_arithmetic_flops=before-arithmetic,saved_fraction=(before-arithmetic)/before,
        unchanged_actor_acquisition_arithmetic_flops=old['actor_acquisition_arithmetic_flops'],
        guidance_extra_arithmetic_flops=arithmetic-old['actor_acquisition_arithmetic_flops'],
        instrumentation_parameter_parity=parity,
        source_sha256={p:e.sha(e.ROOT/p) for p in files},
        input_sha256={**inputs,str(admitted.relative_to(e.ROOT)):e.sha(admitted),str(parent.relative_to(e.ROOT)):e.sha(parent)},
        scope='Executed2048TRAIN preparation with unchanged common actor/source/optimizer and native guidance; full16384deployment parity admitted separately. Compact windows do not need full own/projected endpoints; source actor BPTT remains. Explicit frequency products supplement NumPy proxy. Conventional arithmetic FLOPs, special functions/Python math/random internals separately unpriced; logical accesses not hardware traffic. Preparation saving is not total-lifetime or ASIC energy supremacy.')
    out=e.ROOT/f'experiments/results/market/{args.tag}.json';assert not out.exists()
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source','torch_work','numpy_work','source_sha256','input_sha256']}),flush=True)


if __name__=='__main__':main()
