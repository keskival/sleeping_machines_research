"""B10 TRAIN-mean and signed-native controls for actual prospective updates."""
import argparse,copy,json,time
import market_future_credit_updates as engine
ROOT,torch,np=engine.ROOT,engine.torch,engine.np


class ConstantCredit(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.core=torch.nn.Identity();self.value=value
    def forward(self,start,events,carried):
        return start.new_tensor(self.value)[None],carried


class SignedCredit(torch.nn.Module):
    def __init__(self,critic):
        super().__init__();self.base=copy.deepcopy(critic);self.core=self.base.core
        self.rng=np.random.default_rng(19127)
    def forward(self,start,events,carried):
        output,nextcarry=self.base(start,events,carried)
        signs=start.new_tensor(self.rng.choice([-1.,1.],size=4))[None]
        return output*signs,nextcarry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic()
    model,inputs=engine.acquired.restored()
    parent=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v2_long.json'
    record=json.loads(parent.read_text());inputs[str(parent.relative_to(ROOT))]=engine.acquired.sha(parent)
    checkpoint=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v2_long_critic.pt'
    assert record['input_sha256'][str(checkpoint.relative_to(ROOT))]==engine.acquired.sha(checkpoint)
    packet=torch.load(checkpoint,map_location='cpu',weights_only=False);scale=np.array(packet['scale'])
    critic=engine.acquired.NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,4).double()
    critic.load_state_dict(packet['state_dict']);inputs[str(checkpoint.relative_to(ROOT))]=engine.acquired.sha(checkpoint)
    mean=np.mean([row['target'] for row in record['acquisition']['records']],axis=0)
    t,m,path=engine.acquired.load('2026-08-25',128+16*1024);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    rows=[]
    for name,controller in [('train_mean_refresh',ConstantCredit(mean/scale)),('signed_native_refresh',SignedCredit(critic))]:
        torch.manual_seed(19125)
        row=engine.stream(model,controller,scale,t,m,1024,'learned_refresh');row['arm']=name;rows.append(row)
    result=dict(status='completed',battle='B10',rows=rows,train_mean_credit=mean.tolist(),input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(ROOT/p) for p in (
                    'experiments/credit/market_future_credit_controls.py','experiments/credit/market_future_credit_updates.py')},
                wall_s=time.monotonic()-started,
                scope='One TRAIN-acquired critic/actor seed,16384reusedDEVtargets. TRAIN target mean fixed before DEV; constant control bypasses native computation. Signed control independently flips each native coordinate with seed19127, preserving per-call magnitudes and native work; causal inputs follow its own resulting trajectory. Same clip/Adam and8event refresh as admitted arm. Diagnostic controls, no future DEV teacher, parameter fitting, new reference architecture, TEST or total-work claim.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
