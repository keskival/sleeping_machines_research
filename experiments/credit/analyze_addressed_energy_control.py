"""Source-bound uniform energy-bound control versus native addressed credit."""
import argparse,hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();items=[];inputs={}
    tags=['curie_b10_addressed_clock_20261013_v2_acquisition','curie_b10_addressed_clock_20261013_v3_energy_fit','curie_b10_addressed_clock_20261013_v3_energy_smoke']
    for tag in tags:
        path=ROOT/f'experiments/results/market/{tag}.json';d=json.loads(path.read_text());assert d['status']=='completed'
        for key in ('source_sha256','input_sha256'):
            for file,digest in d.get(key,{}).items():assert sha(ROOT/file)==digest,file
        inputs[str(path.relative_to(ROOT))]=sha(path);items.append(d)
    addressed,uniform,smoke=items;assert addressed['reference']['rows']==uniform['reference']['rows'];assert addressed['acquisition']['targets']==uniform['acquisition']['targets']==8192;assert addressed['dev']['targets']==uniform['dev']['targets']==16384
    assert all(r['paired_zero_action_parity']==0 and r['first_future_parity']<1e-9 for r in smoke['acquisition']['rows'])
    path=ROOT/f'experiments/results/market/{tags[1]}_critic.pt';saved=torch.load(path,map_location='cpu',weights_only=False);entries={k:dict(shape=list(v.shape),dtype=str(v.dtype),values=v.tolist()) for k,v in saved['state_dict'].items()}
    for k,e in entries.items():assert torch.equal(torch.tensor(e['values'],dtype=saved['state_dict'][k].dtype).reshape(e['shape']),saved['state_dict'][k])
    portable=ROOT/f'experiments/results/market/{a.tag}_portable.json'
    with portable.open('x') as f:json.dump(dict(status='completed',checkpoint_sha256=sha(path),scale=saved['scale'],state_dict=entries,scope='Exact fitted tensor roundtrip; no carried DEV state.'),f);f.write('\n')
    inputs[str(portable.relative_to(ROOT))]=sha(portable)
    result=dict(status='completed',battle='B10',parent_nll=uniform['reference']['nll'],addressed_nll=addressed['dev']['nll'],uniform_nll=uniform['dev']['nll'],addressed_parent_gain=addressed['dev_gain'],uniform_parent_gain=uniform['dev_gain'],addressed_gain_over_uniform=uniform['dev']['nll']-addressed['dev']['nll'],parent_forecast_parity=True,uniform_bound=.125,addressed_bound=.5,maximum_residual_l2_bound_equal=True,input_sha256=inputs,source_sha256={'experiments/credit/analyze_addressed_energy_control.py':sha(Path(__file__))},scope='Same actor/parent8192TRAIN256receipts/16384reusedDEV, seeds, paired teacher/radius/optimizer. Matches upper L2 residual bound for equal parent leg scales, not realized updates/Adamstep metric/totalwork. Native uniform critic63608total/31804trainable vs addressed64628/32824; fixed addressed support, not learned selection. Source/teacher/native/optimizer work additional; no TEST/independent confirmation or hardwarewin.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['uniform_nll'],result['addressed_nll'],result['addressed_gain_over_uniform'],flush=True)
if __name__=='__main__':main()
