"""B10 acquisition curve at fixed data with reused forward-teacher receipts."""
import argparse,json,time
import market_future_interface_scaled as base
ROOT,torch,np=base.ROOT,base.torch,base.np


def packets(model,t,m,targets):
    state=base.warm(model,t,m,128);rows=[]
    with torch.no_grad():
        for block,target in enumerate(targets):
            k=127+16*block;old=base.flat(state)
            scores=[]
            for index in range(k,k+8):
                forecast,state=base.control.market_consume(model,t[index:index+1],m[index:index+1],state)
                scores.append(float(-base.control.revealed_log_probability(model,forecast,t[index+1:index+2],m[index+1:index+2]).mean()))
            current=base.flat(state);features=torch.cat((old,current,current-old))[None]
            feedback=current.new_zeros(8)
            feedback[0]=torch.log1p((t[k+8]-t[k])/model.scale)
            feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+9],6).double().mean(0)
            feedback[7]=float(np.mean(scores))
            rows.append((features,torch.cat((features[0],feedback))[None,None],torch.tensor(target)[None]))
            _,state=base.future(model,state,t,m,k+8,8);state=base.control.detach_state(state)
    return rows


def replay(critic,optimizer,rows,scale,training):
    carry=None;predictions=[];begin=time.monotonic();critic.core.train(training)
    for features,events,target in rows:
        with torch.set_grad_enabled(training):prediction,carry=critic(features,events,carry)
        predictions.append((prediction.detach().numpy()[0]*scale).tolist())
        if training:
            optimizer.zero_grad(set_to_none=True)
            (prediction-target/torch.as_tensor(scale)[None]).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    truth=np.array([target.numpy()[0] for _,_,target in rows]);p=np.array(predictions)
    return dict(predictions=predictions,mse=float(np.mean((p-truth)**2)),zero_mse=float(np.mean(truth**2)),
                wall_s=time.monotonic()-begin,frames=len(rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19124);started=time.monotonic()
    parent=ROOT/'experiments/results/market/curie_b10_future_interface_scaled_20261013_v1_fit.json'
    source=json.loads(parent.read_text());model,inputs=base.restored();inputs[str(parent.relative_to(ROOT))]=base.sha(parent)
    scale=np.array(source['scale']);build_start=time.monotonic();cached={}
    for phase,date in [('train','2026-08-24'),('dev','2026-08-25')]:
        targets=[row['target'] for row in source[phase]['records']]
        t,m,path=base.load(date,128+16*len(targets));inputs[str(path.relative_to(ROOT))]=base.sha(path)
        cached[phase]=packets(model,t,m,targets)
    preparation=time.monotonic()-build_start
    critic=base.NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,4).double()
    torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
    optimizer=torch.optim.Adam(critic.parameters(),lr=.001);curve=[];first_error=None
    for epoch in range(1,5):
        train=replay(critic,optimizer,cached['train'],scale,True)
        rng=torch.get_rng_state()
        dev=replay(critic,optimizer,cached['dev'],scale,False)
        torch.set_rng_state(rng) # evaluation cannot change subsequent fitting RNG
        if epoch==1:
            first_error=max(abs(a-b) for row,ref in zip(train['predictions'],source['train']['records']) for a,b in zip(row,ref['prediction']))
            assert first_error==0
            assert dev['mse']==source['dev']['mse']
        curve.append(dict(epoch=epoch,train=train,dev=dev))
    checkpoint=ROOT/f'experiments/results/market/{args.tag}_critic.pt'
    assert not checkpoint.exists();torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist()),checkpoint)
    inputs[str(checkpoint.relative_to(ROOT))]=base.sha(checkpoint)
    result=dict(status='completed',battle='B10',curve=curve,first_pass_prediction_error=first_error,
                unique_train_frames=128,unique_train_targets=2048,original_forward_teacher_targets=8192,
                additional_teacher_worlds=0,native_fit_frames=512,packet_preparation_s=preparation,
                wall_s=time.monotonic()-started,input_sha256=inputs,
                source_sha256={p:base.sha(ROOT/p) for p in ('experiments/credit/market_future_credit_reuse.py',
                    'experiments/credit/market_future_interface_scaled.py','experiments/credit/native_market_credit.py')},
                scope='Fixed-data acquisition curve:128TRAINreceipts,4chronological native fits with episode bank reset/pass and weights/moments retained;32DEVreceipts evaluated frozen after everypass. Same forward labels/current actor version; no new teacherworlds or unique examples. DEV checkpoints are development diagnostics, not sealed selection. First-pass acquired predictions and DEV MSE exactly reproduce retained model. More native fitting/key/value/optimizer work charged; original8192teacher targets and actor fit additional. No actor-update quality/data-to-target or matched-compute/energy claim from credit MSE alone.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in curve:print(row['epoch'],row['dev']['mse'],row['dev']['zero_mse'],flush=True)


if __name__=='__main__':main()
