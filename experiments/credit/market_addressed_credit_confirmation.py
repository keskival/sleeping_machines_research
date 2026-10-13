"""B10 unchanged addressed/uniform energy-bound confirmation on saved parents."""
import argparse,copy,json,time
import market_addressed_clock_credit_acquisition as routed
import market_energy_matched_uniform_fit as uniform
from native_addressed_clock_credit import AddressedClockCredit,guided
from energy_matched_uniform_credit import EnergyMatchedUniformCredit
ROOT,torch,np,engine=routed.ROOT,routed.torch,routed.np,routed.engine

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--parent-seed',type=int,choices=(19129,19130),required=True);p.add_argument('--member',choices=('addressed','uniform'),required=True);p.add_argument('--mode',choices=('smoke','fit'),default='fit');a=p.parse_args();frames=3 if a.mode=='smoke' else 256;blocks=3 if a.mode=='smoke' else 1024;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19131);begin=time.monotonic()
    routed.teacher.guided=guided;uniform.teacher.guided=guided
    model,inputs=engine.acquired.restored();original=copy.deepcopy(model)
    path=ROOT/f'experiments/results/market/curie_b10_future_credit_confirm_20261013_s{a.parent_seed}_v1_critic.pt';packet=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    parent=routed.native.NativeMarketCredit(264).double();parent.core.head=torch.nn.Linear(parent.core.total_payload,4).double();parent.load_state_dict(packet['state_dict']);parent_scale=np.array(packet['scale'])
    if a.member=='addressed':critic=AddressedClockCredit(parent,parent_scale);scale=np.repeat(parent_scale,16)/4;fit=routed.fit
    else:critic=EnergyMatchedUniformCredit(parent,parent_scale);scale=parent_scale;fit=uniform.fit
    t,m,path=engine.acquired.load('2026-08-24',128+32*frames);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path);critic,source=fit(model,critic,scale,t,m,frames,a.mode=='smoke')
    t,m,path=engine.acquired.load('2026-08-25',128+16*blocks);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    torch.manual_seed(19125);_,dev=routed.native.stream(original,critic,parent,scale,t,m,blocks,False)
    torch.manual_seed(19125);reference=engine.stream(original,parent,parent_scale,t,m,blocks,'learned_refresh')
    expected_path=ROOT/'experiments/results/market/curie_b10_future_credit_confirmation_20261013_v1.json';expected=json.loads(expected_path.read_text());inputs[str(expected_path.relative_to(ROOT))]=engine.acquired.sha(expected_path)
    saved=next(row['nll'] for row in expected['rows'] if row['critic_seed']==a.parent_seed);assert a.mode=='smoke' or abs(reference['nll']-saved)<1e-12
    names=['experiments/credit/market_addressed_credit_confirmation.py','experiments/credit/market_addressed_clock_credit_acquisition.py','experiments/credit/market_energy_matched_uniform_fit.py','experiments/credit/native_addressed_clock_credit.py','experiments/credit/energy_matched_uniform_credit.py','experiments/credit/native_optimizer_receipts.py','experiments/credit/native_receipt_evaluation.py','experiments/credit/market_actual_update_teacher.py']
    result=dict(status='completed',battle='B10',args=vars(a),acquisition=source,dev=dev,reference=reference,dev_gain=reference['nll']-dev['nll'],reference_saved_error=abs(reference['nll']-saved) if a.mode=='fit' else None,input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in names},wall_s=time.monotonic()-begin,scope='Fixed256receipts8192TRAIN/16384reusedDEV member, saved independently initialized parent critic, same restoredactor and causal score-before-update protocol. Uniform0.125 matches addressed0.5 selected-mode maximum residual L2bound; actualsteps/headparameter counts differ. Two extra forward worlds/receipt, native/privatebank and actor/optimizer costs charged. Criticweightsfrozen onDEV; no TEST/independentactor/data replication/fullwork claim.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist()),ROOT/f'experiments/results/market/{a.tag}_critic.pt')
    print(a.parent_seed,a.member,dev['nll'],reference['nll'],result['dev_gain'],flush=True)
if __name__=='__main__':main()
