"""B10 null-path parity and optimizer-cut receipt distinguishability."""
import argparse,copy,json,time
import native_receipt_evaluation as evaluation
import native_optimizer_receipts as receipts
import native_addressed_clock_credit as addressed
teacher_guided_placeholder=None
from native_market_credit import NativeMarketCredit
ROOT,torch,np,engine,teacher=evaluation.ROOT,evaluation.torch,evaluation.np,evaluation.engine,evaluation.teacher

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    model,inputs=engine.acquired.restored();path=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt';packet=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    parent=NativeMarketCredit(264).double();parent.core.head=torch.nn.Linear(parent.core.total_payload,4).double();parent.load_state_dict(packet['state_dict']);scale=np.array(packet['scale']);critic=addressed.AddressedClockCredit(parent,scale);teacher.guided=addressed.guided
    assert torch.all(torch.stack([addressed.support(i) for i in range(32)]).sum(0)==2)
    t,m,path=engine.acquired.load('2026-08-25',128+48);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    torch.manual_seed(19125);_,actual=evaluation.stream(model,critic,parent,np.repeat(scale,16)/4,t,m,3,False);end_rng=torch.get_rng_state().clone()
    torch.manual_seed(19125);reference=engine.stream(model,parent,scale,t,m,3,'learned_refresh');assert torch.equal(torch.get_rng_state(),end_rng)
    error=max(abs(x[k]-y[k]) for x,y in zip(actual['rows'],reference['rows']) for k in ('producer_nll','future_nll'));assert error==0
    opt=torch.optim.Adam(model.parameters(),lr=.0003);state=engine.acquired.warm(model,t,m,128);alias_rows=[]
    for block in range(3):
        k=127+16*block;old=engine.acquired.control.detach_state(state);end,grads,scores=engine.window(model,state,t,m,k,8,'own')
        first,last=engine.acquired.flat(old),engine.acquired.flat(end);features=torch.cat((first,last,last-first))[None];observed=torch.cat((features,features.new_zeros(1,8)),-1)[:,None]
        if block:
            other_model,other_opt=copy.deepcopy((model,opt))
            for name,p in other_model.named_parameters():
                if name.endswith(('log_rate','freq')):
                    for key in ('exp_avg','exp_avg_sq'):other_opt.state[p][key]=other_opt.state[p][key].flip(0).clone()
            base=features.new_tensor([.003,-.008,.002,-.004]);sc=features.new_ones(4)
            one=receipts.events(features,observed,receipts.descriptor(model,opt,grads),base,sc)
            two=receipts.events(features,observed,receipts.descriptor(other_model,other_opt,grads),base,sc)
            difference=float((one-two).abs().max());assert difference>1e-10
            alias_rows.append(dict(block=block,token_max_difference=difference,node_tokens=one.shape[1]-1))
        engine.step(model,opt,teacher.guided(model,grads,features.new_tensor([.003,-.008,.002,-.004])))
        with torch.no_grad():
            state=old
            for index in range(k,k+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    names=['experiments/credit/check_addressed_clock_credit.py','experiments/credit/native_addressed_clock_credit.py','experiments/credit/native_optimizer_receipts.py','experiments/credit/native_receipt_evaluation.py']
    d=dict(status='completed',battle='B10',null_forecast_error=error,null_factual_rng_parity=True,available_clock_coordinates=64,selected_clock_coordinates=4,support_union_coordinates=64,alias_rows=alias_rows,total_parameters=sum(p.numel() for p in critic.parameters()),trainable_parameters=sum(p.numel() for p in critic.parameters() if p.requires_grad),input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in names},wall_s=time.monotonic()-begin,scope='Three48DEVtarget null-path updates and two momentpermutation cuts. Zero residual reproduces parent forecasts/factualrace RNG exactly;32mode tokens distinguish former pooled-context aliases. No fittedquality/hardwarewin. Private native clocks/newbanks/parentcapacity paid; macroglobalclip retained.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(d,flush=True)
if __name__=='__main__':main()
