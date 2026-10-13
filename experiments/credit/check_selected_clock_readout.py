"""B10 trained native selected-readout output, gradient and activity parity."""
import argparse,copy,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import native_addressed_clock_credit as addressed
import selected_clock_readout as selected
import market_actual_utility_fit_v2 as native
from native_optimizer_receipts import descriptor
ROOT,torch,np,engine=native.ROOT,native.torch,native.np,native.engine

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    model,inputs=engine.acquired.restored();original=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt';packet=torch.load(original,map_location='cpu',weights_only=False);inputs[str(original.relative_to(ROOT))]=engine.acquired.sha(original)
    parent=native.NativeMarketCredit(264).double();parent.core.head=torch.nn.Linear(parent.core.total_payload,4).double();parent.load_state_dict(packet['state_dict']);critic=addressed.AddressedClockCredit(parent,packet['scale'])
    checkpoint=ROOT/'experiments/results/market/curie_b10_addressed_clock_20261013_v2_acquisition_critic.pt';trained=torch.load(checkpoint,map_location='cpu',weights_only=False);inputs[str(checkpoint.relative_to(ROOT))]=engine.acquired.sha(checkpoint);critic.load_state_dict(trained['state_dict']);compiled=selected.compile_readout(critic)
    opt=torch.optim.Adam(model.parameters(),lr=.0003);t,m,path=engine.acquired.load('2026-08-24',128+8*4);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path);state=engine.acquired.warm(model,t,m,128);one_carry=two_carry=None;rows=[];calls=[]
    hook=critic.core.head.register_forward_hook(lambda module,args,out:calls.append(1))
    for block in range(4):
        k=127+8*block;old=engine.acquired.control.detach_state(state);state,grads,scores=engine.window(model,state,t,m,k,8,'own');first,last=engine.acquired.flat(old),engine.acquired.flat(state);features=torch.cat((first,last,last-first))[None]
        feedback=last.new_zeros(8);feedback[0]=torch.log1p((t[k+8]-t[k])/model.scale);feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+9],6).double().mean(0);feedback[7]=float(np.mean(scores));events=torch.cat((features[0],feedback))[None,None];operator=descriptor(model,opt,grads)
        issue_rng=torch.get_rng_state().clone();out,one_carry=critic(features,events,one_carry,operator);end_rng=torch.get_rng_state().clone();torch.set_rng_state(issue_rng);other,two_carry=compiled(features,events,two_carry,operator);assert torch.equal(torch.get_rng_state(),end_rng);assert torch.equal(one_carry.rng,two_carry.rng)
        error=float((out-other).abs().max());weights=torch.linspace(-.5,.5,64)[None];one=torch.autograd.grad((out*weights).sum(),[v for v in critic.parameters() if v.requires_grad],allow_unused=True);two=torch.autograd.grad((other*weights).sum(),[v for v in compiled.parameters() if v.requires_grad],allow_unused=True);assert all((x is None)==(y is None) for x,y in zip(one,two));gradient_error=max(float((x-y).abs().max()) for x,y in zip(one,two) if x is not None);assert max(error,gradient_error)<1e-12
        rows.append(dict(block=block,output_error=error,gradient_error=gradient_error,factual_rng_parity=True,selected_indices=compiled.core.head.indices.tolist()));state=engine.acquired.control.detach_state(state)
    hook.remove();assert len(calls)==compiled.core.head.calls;count=len(calls);width=critic.core.total_payload
    result=dict(status='completed',battle='B10',rows=rows,readout_calls=count,dense_rows=64,selected_rows=4,readout_width=width,dense_readout_flops=count*(2*64*width+64),selected_readout_flops=count*(2*4*width+4),dense_weight_bias_scalar_reads=count*64*(width+1),selected_weight_bias_scalar_reads=count*4*(width+1),input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in ['experiments/credit/selected_clock_readout.py','experiments/credit/check_selected_clock_readout.py','experiments/credit/native_addressed_clock_credit.py']},wall_s=time.monotonic()-begin,scope='Four actual native receipt decisions with fitted256receipt critic, output/VJP and pairedglobalRNG checks. Matmul2MAC plusbias convention, logical weight/bias operands only:16x lower readout component work/reads. Gather/scatter/64output allocation, nonlinear/core/memory/history/actor/optimizer/teacher work excluded; no full-fit physicaltraffic/energy/qualitywin.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
