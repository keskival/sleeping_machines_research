"""B10 four-pass acquired critic, unchanged actual-update/scoring protocol."""
import argparse,json,time
import market_future_credit_updates as engine


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    engine.torch.set_num_threads(1);engine.torch.set_default_dtype(engine.torch.float64);started=time.monotonic()
    root=engine.ROOT;model,inputs=engine.acquired.restored()
    parent=root/'experiments/results/market/curie_b10_future_credit_reuse_20261013_v1_fit.json'
    record=json.loads(parent.read_text());inputs[str(parent.relative_to(root))]=engine.acquired.sha(parent)
    checkpoint=root/'experiments/results/market/curie_b10_future_credit_reuse_20261013_v1_fit_critic.pt'
    assert record['input_sha256'][str(checkpoint.relative_to(root))]==engine.acquired.sha(checkpoint)
    packet=engine.torch.load(checkpoint,map_location='cpu',weights_only=False);scale=engine.np.array(packet['scale'])
    critic=engine.acquired.NativeMarketCredit(264).double();critic.core.head=engine.torch.nn.Linear(critic.core.total_payload,4).double()
    critic.load_state_dict(packet['state_dict']);inputs[str(checkpoint.relative_to(root))]=engine.acquired.sha(checkpoint)
    reference=root/'experiments/results/market/curie_b10_future_updates_20261013_v2_long.json'
    inputs[str(reference.relative_to(root))]=engine.acquired.sha(reference)
    t,m,path=engine.acquired.load('2026-08-25',128+16*1024);inputs[str(path.relative_to(root))]=engine.acquired.sha(path)
    engine.torch.manual_seed(19125)
    row=engine.stream(model,critic,scale,t,m,1024,'learned_refresh');row['arm']='four_pass_learned_refresh'
    result=dict(status='completed',battle='B10',rows=[row],input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(root/p) for p in ('experiments/credit/market_future_credit_reuse_updates.py',
                    'experiments/credit/market_future_credit_updates.py','experiments/credit/market_future_credit_reuse.py')},
                wall_s=time.monotonic()-started,
                scope='Fixed128TRAINreceipts/2048uniqueTRAINtargets,4nativecriticpasses instead of1; no additional teacherworlds. Identical savedactor/freshAdam/scales/producer8-future8/refresh/additivecoefficient1 and16384DEVtargets as completed controls. More acquisition work charged. Frozencriticweights/privateepisodehistorycausal; actor updates online. One initialization/reusedDEVacquisitioncurve checkpoints, no newTEST/data-to-quality/materialefficiency/energy claim.')
    with (root/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
