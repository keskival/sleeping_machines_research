"""B10 unchanged fitted action-geometry critic: full causal DEV extension."""
import argparse,copy,json,time
import market_actual_utility_fit_v2 as fit
ROOT,torch,np,engine=fit.ROOT,fit.torch,fit.np,fit.engine

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    model,inputs=engine.acquired.restored()
    parent=ROOT/'experiments/results/market/curie_b10_delayed_action_20261013_v3_large.json';record=json.loads(parent.read_text());inputs[str(parent.relative_to(ROOT))]=engine.acquired.sha(parent)
    original=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt';packet=torch.load(original,map_location='cpu',weights_only=False);inputs[str(original.relative_to(ROOT))]=engine.acquired.sha(original)
    anchor=fit.NativeMarketCredit(264).double();anchor.core.head=torch.nn.Linear(anchor.core.total_payload,4).double();anchor.load_state_dict(packet['state_dict'])
    critic=fit.UtilityCredit(copy.deepcopy(anchor));path=ROOT/'experiments/results/market/curie_b10_delayed_action_20261013_v3_large_critic.pt';trained=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path);critic.load_state_dict(trained['state_dict']);scale=np.array(trained['scale'])
    t,m,path=engine.acquired.load('2026-08-25',128+16384);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    torch.manual_seed(19125);_,dev=fit.stream(model,critic,anchor,scale,t,m,1024,False)
    error=max(abs(row[key]-old[key]) for row,old in zip(dev['rows'][:128],record['dev']['rows']) for key in ('producer_nll','future_nll'));assert error==0
    torch.manual_seed(19125);reference=engine.stream(model,anchor,scale,t,m,1024,'learned_refresh')
    result=dict(status='completed',battle='B10',dev=dev,reference=reference,dev_gain=reference['nll']-dev['nll'],prefix_parity=error,input_sha256=inputs,source_sha256={p:engine.acquired.sha(ROOT/p) for p in ('experiments/credit/market_delayed_action_full.py','experiments/credit/market_actual_utility_fit_v2.py','experiments/credit/market_delayed_action_teacher_weighted_large.py')},wall_s=time.monotonic()-begin,scope='Unchanged saved256TRAINdelayedactioncritic, Cweightsfrozen/causalprivatebanks,16384reusedDEVevents, sameactor/freshAdam/routeRNG8+8 protocol. No additional source fit/teacher atDEV; source8192actualTRAINevents plus256finite alternative learner worlds charged, exact2048prefixparity. Singlecritic/actor, original32targetleadingprotocolunchanged; no TEST/profit/hardwareclaim.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(dev['nll'],reference['nll'],result['dev_gain'],flush=True)
if __name__=='__main__':main()
