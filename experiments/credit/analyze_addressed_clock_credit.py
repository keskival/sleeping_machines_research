"""Source-bound addressed-clock acquisition curve and complete DEV decision."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();items={};inputs={}
    tags=['curie_b10_addressed_clock_20261013_v1_contract','curie_b10_addressed_clock_20261013_v1_smoke','curie_b10_addressed_clock_20261013_v1_fit','curie_b10_addressed_clock_20261013_v2_acquisition']
    for tag in tags:
        path=ROOT/f'experiments/results/market/{tag}.json';d=json.loads(path.read_text());assert d['status']=='completed'
        for key in ('source_sha256','input_sha256'):
            for file,digest in d.get(key,{}).items():assert sha(ROOT/file)==digest,file
        inputs[str(path.relative_to(ROOT))]=sha(path);items[tag]=d
    contract,smoke,small,large=[items[t] for t in tags]
    assert contract['null_forecast_error']==0 and contract['null_factual_rng_parity'] and contract['support_union_coordinates']==64
    assert all(r['paired_zero_action_parity']==0 and r['first_future_parity']<1e-9 for r in smoke['acquisition']['rows'])
    assert large['acquisition']['rows'][:64]==small['acquisition']['rows']
    assert large['reference']['rows'][:128]==small['reference']['rows']
    curve=[]
    for tag,d in [(tags[2],small),(tags[3],large)]:
        prefix=statistics.mean(.5*(r['producer_nll']+r['future_nll']) for r in d['dev']['rows'][:128]);ref=statistics.mean(.5*(r['producer_nll']+r['future_nll']) for r in d['reference']['rows'][:128])
        curve.append(dict(tag=tag,receipts=d['acquisition']['frames'],train_events=d['acquisition']['targets'],supervised_decisions_per_mode=d['acquisition']['frames']//16,prefix_nll=prefix,prefix_parent_nll=ref,prefix_gain=ref-prefix))
        path=ROOT/f'experiments/results/market/{tag}_critic.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        entries={key:dict(shape=list(v.shape),dtype=str(v.dtype),values=v.tolist()) for key,v in saved['state_dict'].items()}
        for key,e in entries.items():assert torch.equal(torch.tensor(e['values'],dtype=saved['state_dict'][key].dtype).reshape(e['shape']),saved['state_dict'][key])
        portable=ROOT/f'experiments/results/market/{tag}_portable.json'
        with portable.open('x') as f:json.dump(dict(status='completed',checkpoint_sha256=sha(path),scale=saved['scale'],state_dict=entries,scope='Exact fitted tensors; no carried DEV state.'),f);f.write('\n')
        inputs[str(portable.relative_to(ROOT))]=sha(portable)
    result=dict(status='completed',battle='B10',curve=curve,source_prefix_parity=True,parent_prefix_parity=True,full_targets=large['dev']['targets'],full_nll=large['dev']['nll'],full_parent_nll=large['reference']['nll'],full_gain=large['dev_gain'],quarter_gains=[statistics.mean(.5*(x['producer_nll']+x['future_nll']-y['producer_nll']-y['future_nll']) for x,y in zip(large['reference']['rows'][k:k+256],large['dev']['rows'][k:k+256])) for k in range(0,1024,256)],input_sha256=inputs,source_sha256={'experiments/credit/analyze_addressed_clock_credit.py':sha(Path(__file__))},decision='Retain parent unless completed full trajectory improves; improvement remains one-actor development evidence.',scope='64/256receipts2048/8192TRAIN; unchanged member, exact first64source and parent DEV prefixes. Large model scored16384reusedDEV, native weights frozen/causal private memory. Four residual targets per decision,64available and all64readout values computed; old parent/own gradients still cover all clock coordinates.512extra worlds12288score/8192refresh targets,4096own-gradient targets within scoring; parent/actor acquisition/native work additional. No TEST/ASIC claim.')
    assert result['full_targets']==16384
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['curve','full_nll','full_parent_nll','full_gain','quarter_gains']}),flush=True)
if __name__=='__main__':main()
