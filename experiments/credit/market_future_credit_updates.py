"""B10 actual Adam with forward-taught coordinate credit and causal refresh."""
import argparse,copy,json,time
import market_future_interface_scaled as acquired
from market_streaming_clock_updates import window
from market_native_remote_admission_selected import step

ROOT,torch,np=acquired.ROOT,acquired.torch,acquired.np


def stream(model,critic,scale,t,m,blocks,arm):
    model=copy.deepcopy(model);critic=copy.deepcopy(critic)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    state=acquired.warm(model,t,m,128);carry=None;rows=[];begin=time.monotonic()
    phase={key:0. for key in ('own_credit','critic','refresh','future','optimizer')}
    norms=[];refresh_targets=0
    critic.core.train(False)
    for block in range(blocks):
        k=127+16*block;old=acquired.control.detach_state(state)
        tick=time.monotonic()
        end,grads,scores=window(model,state,t,m,k,8,'own')
        phase['own_credit']+=time.monotonic()-tick
        prediction=np.zeros(4)
        if arm=='learned_refresh':
            tick=time.monotonic();first,last=acquired.flat(old),acquired.flat(end)
            features=torch.cat((first,last,last-first))[None]
            feedback=last.new_zeros(8)
            feedback[0]=torch.log1p((t[k+8]-t[k])/model.scale)
            feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+9],6).double().mean(0)
            feedback[7]=float(np.mean(scores))
            with torch.no_grad():output,carry=critic(features,torch.cat((features[0],feedback))[None,None],carry)
            prediction=output.numpy()[0]*scale
            names=[name for name,_ in model.named_parameters()]
            for j,layer in enumerate(model.layers):
                for offset,key in enumerate(('log_rate','freq')):
                    grads[names.index(f'layers.{j}.{key}')]+=float(prediction[2*j+offset])/np.sqrt(layer.n)
            phase['critic']+=time.monotonic()-tick
        norms.append(float(np.linalg.norm(prediction)))
        tick=time.monotonic();step(model,optimizer,grads);phase['optimizer']+=time.monotonic()-tick
        if arm.endswith('refresh'):
            tick=time.monotonic()
            with torch.no_grad():
                refreshed=old
                for index in range(k,k+8):
                    _,refreshed=acquired.control.market_consume(model,t[index:index+1],m[index:index+1],refreshed)
            end=acquired.control.detach_state(refreshed);refresh_targets+=8
            phase['refresh']+=time.monotonic()-tick
        tick=time.monotonic()
        future_scores=[]
        with torch.no_grad():
            state=end
            for index in range(k+8,k+16):
                forecast,state=acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
                future_scores.append(float(-acquired.control.revealed_log_probability(model,forecast,t[index+1:index+2],m[index+1:index+2]).mean()))
        state=acquired.control.detach_state(state);phase['future']+=time.monotonic()-tick
        rows.append(dict(producer_nll=float(np.mean(scores)),future_nll=float(np.mean(future_scores))))
    return dict(arm=arm,targets=16*blocks,updates=blocks,nll=float(np.mean([.5*(r['producer_nll']+r['future_nll']) for r in rows])),
                future_nll=float(np.mean([r['future_nll'] for r in rows])),rows=rows,
                causal_refresh_targets=refresh_targets,teacher_worlds=0,
                critic_prediction_norm_mean=float(np.mean(norms)),wall_s=time.monotonic()-begin,phase_wall_s=phase)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--blocks',type=int,default=128);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19124)
    begin=time.monotonic();model,inputs=acquired.restored()
    t,m,path=acquired.load('2026-08-24',128+16*128);inputs[str(path.relative_to(ROOT))]=acquired.sha(path)
    state=acquired.warm(model,t,m,128);checked=acquired.contract(model,t,m,state,127)
    critic=acquired.NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,4).double()
    torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
    optimizer=torch.optim.Adam(critic.parameters(),lr=.001)
    scale=np.maximum(1e-4,np.abs(np.array(checked['exact_directional_credit'])))
    source=acquired.collect(model,critic,optimizer,t,m,state,128,True,scale)
    parent=ROOT/'experiments/results/market/curie_b10_future_interface_scaled_20261013_v1_fit.json'
    reference=json.loads(parent.read_text());inputs[str(parent.relative_to(ROOT))]=acquired.sha(parent)
    source_error=max(abs(a-b) for new,old in zip(source['records'],reference['train']['records']) for a,b in zip(new['prediction'],old['prediction']))
    assert source_error==0
    td,md,path=acquired.load('2026-08-25',128+16*args.blocks);inputs[str(path.relative_to(ROOT))]=acquired.sha(path)
    rows=[]
    # Independent reset of native route RNG for every chronological arm.
    for arm in ('own_carry','own_refresh','learned_refresh'):
        torch.manual_seed(19125)
        rows.append(stream(model,critic,scale,td,md,args.blocks,arm))
    checkpoint=ROOT/f'experiments/results/market/{args.tag}_critic.pt'
    assert not checkpoint.exists();torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist(),actor_inputs=inputs),checkpoint)
    inputs[str(checkpoint.relative_to(ROOT))]=acquired.sha(checkpoint)
    result=dict(status='completed',battle='B10',args=vars(args),rows=rows,acquisition=source,
                acquisition_prediction_replay_error=source_error,contract=checked,
                source_sha256={p:acquired.sha(ROOT/p) for p in (
                    'experiments/credit/market_future_credit_updates.py','experiments/credit/market_future_interface_scaled.py',
                    'experiments/credit/market_streaming_clock_updates.py','experiments/market/local_eligibility_state.py',
                    'experiments/credit/native_market_credit.py','experiments/market/episode_state.py')},
                input_sha256=inputs,wall_s=time.monotonic()-begin,
                scope='One actor/critic seed, reused chronological DEV, producer8/future8 targets, actual clip1/Adam.0003 updates after producer. Native weights frozen, private history evolves; predicted four-coordinate future-state credit added to own gradients. Refresh replays observed producer events from detached pre-producer state under new weights,8extra forward events/update. Own-refresh isolates predicted-credit utility; own-carry isolates refresh. Actor updates change distribution from fixed-actor acquisition. No future teacher on deployment, BPTT imitation, sealed TEST, complete-work or energy claim;128-frame TRAIN acquisition8192teacher targets and original actor fitting additionally charged.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','acquisition','input_sha256','source_sha256')}),flush=True)
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
