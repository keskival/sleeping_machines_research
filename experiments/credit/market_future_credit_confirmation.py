"""B10 independent critic initialization, fixed data/actual-update protocol."""
import argparse,json,time
import market_future_credit_reuse as reuse
import market_future_credit_updates as engine
ROOT,torch,np=engine.ROOT,engine.torch,engine.np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(args.seed);started=time.monotonic()
    parent=ROOT/'experiments/results/market/curie_b10_future_interface_scaled_20261013_v1_fit.json'
    labels=json.loads(parent.read_text());model,inputs=engine.acquired.restored();inputs[str(parent.relative_to(ROOT))]=engine.acquired.sha(parent)
    scale=np.array(labels['scale']);cache={}
    for phase,date in [('train','2026-08-24'),('dev','2026-08-25')]:
        t,m,path=engine.acquired.load(date,128+16*len(labels[phase]['records']));inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
        cache[phase]=reuse.packets(model,t,m,[r['target'] for r in labels[phase]['records']])
    critic=engine.acquired.NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,4).double()
    torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
    optimizer=torch.optim.Adam(critic.parameters(),lr=.001)
    train=reuse.replay(critic,optimizer,cache['train'],scale,True)
    dev=reuse.replay(critic,optimizer,cache['dev'],scale,False)
    t,m,path=engine.acquired.load('2026-08-25',128+16*1024);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    torch.manual_seed(19125);row=engine.stream(model,critic,scale,t,m,1024,'learned_refresh')
    checkpoint=ROOT/f'experiments/results/market/{args.tag}_critic.pt'
    assert not checkpoint.exists();torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist()),checkpoint)
    inputs[str(checkpoint.relative_to(ROOT))]=engine.acquired.sha(checkpoint)
    result=dict(status='completed',battle='B10',args=vars(args),train=train,dev=dev,rows=[row],input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(ROOT/p) for p in ('experiments/credit/market_future_credit_confirmation.py',
                    'experiments/credit/market_future_credit_reuse.py','experiments/credit/market_future_credit_updates.py')},
                wall_s=time.monotonic()-started,
                scope='Independent critic initialization, same restoredactor/128TRAINforwardreceipts/1pass/16384DEVupdate protocol and fixedscales. Shared original actor fit and8192teacher targets paid; reuse adds no teacherworlds. Actor initialweights/moments/commondata unchanged, native weightsfrozen duringDEV/causal banks evolve; no independentactor/data replicate or newTEST/energy/data-to-quality claim. Compare saved deterministic own/BP/meancontrols; originalsignednativecontrol belongs to originalcritic only.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(dev['mse'],row['nll'],row['future_nll'],flush=True)


if __name__=='__main__':main()
