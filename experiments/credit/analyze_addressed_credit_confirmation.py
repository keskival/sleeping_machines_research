"""Paired three-parent native addressed-credit cohort and portable evidence."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(tag,inputs):
    path=ROOT/f'experiments/results/market/{tag}.json';d=json.loads(path.read_text());assert d['status']=='completed'
    for key in ['source_sha256','input_sha256']:
        for file,digest in d.get(key,{}).items():assert sha(ROOT/file)==digest,file
    inputs[str(path.relative_to(ROOT))]=sha(path);return d

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();inputs={};rows=[]
    for seed in [19124,19129,19130]:
        if seed==19124:one='curie_b10_addressed_clock_20261013_v2_acquisition';two='curie_b10_addressed_clock_20261013_v3_energy_fit'
        else:
            stem=f'curie_b10_addressed_clock_20261013_confirm_s{seed}_v1';one=stem+'_addressed';two=stem+'_uniform'
        addressed=load(one,inputs);uniform=load(two,inputs);assert addressed['reference']['rows']==uniform['reference']['rows'];assert addressed['dev']['targets']==uniform['dev']['targets']==16384;assert addressed['acquisition']['targets']==uniform['acquisition']['targets']==8192
        rows.append(dict(parent_seed=seed,parent_nll=addressed['reference']['nll'],addressed_nll=addressed['dev']['nll'],uniform_nll=uniform['dev']['nll'],parent_gain=addressed['dev_gain'],uniform_gain=uniform['dev']['nll']-addressed['dev']['nll']))
        if seed!=19124:
            for tag in [one,two]:
                path=ROOT/f'experiments/results/market/{tag}_critic.pt';saved=torch.load(path,map_location='cpu',weights_only=False);entries={k:dict(shape=list(v.shape),dtype=str(v.dtype),values=v.tolist()) for k,v in saved['state_dict'].items()}
                for k,e in entries.items():assert torch.equal(torch.tensor(e['values'],dtype=saved['state_dict'][k].dtype).reshape(e['shape']),saved['state_dict'][k])
                portable=ROOT/f'experiments/results/market/{tag}_portable.json'
                with portable.open('x') as f:json.dump(dict(status='completed',checkpoint_sha256=sha(path),scale=saved['scale'],state_dict=entries,scope='Exact tensor roundtrip, no carriedDEVstate.'),f);f.write('\n')
                inputs[str(portable.relative_to(ROOT))]=sha(portable)
    result=dict(status='completed',battle='B10',rows=rows,mean_nll=statistics.mean(r['addressed_nll'] for r in rows),sample_sd_nll=statistics.stdev(r['addressed_nll'] for r in rows),mean_parent_gain=statistics.mean(r['parent_gain'] for r in rows),mean_uniform_gain=statistics.mean(r['uniform_gain'] for r in rows),all_parent_wins=all(r['parent_gain']>0 for r in rows),all_uniform_wins=all(r['uniform_gain']>0 for r in rows),input_sha256=inputs,source_sha256={'experiments/credit/analyze_addressed_credit_confirmation.py':sha(Path(__file__))},scope='Three saved independently initialized native parents, one sharedactor/TRAIN/DEV, one fixedresidual trainingseed/radius. Paired maximum-energy-bound controls, no tuning between members/parents. Eachmember8192TRAIN/16384DEV with512extra worlds12288score+8192refresh;4096own-gradient targets within scoring. Headparameter counts and realized updates differ. No TEST/independentactor/data replication/totalwork/ASICenergy claim.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['rows','mean_parent_gain','mean_uniform_gain','all_parent_wins','all_uniform_wins']}),flush=True)
if __name__=='__main__':main()
