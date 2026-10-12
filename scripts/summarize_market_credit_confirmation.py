"""B10 fixed-seed continuation decision from completed immutable results."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();rows=[];parents={}
 for seed in [184,185,186]:
  tag='curie_market_future_shift_20261012_v1_fit' if seed==184 else f'curie_market_future_shift_20261012_v1_s{seed}_fit'
  p=ROOT/f'experiments/results/credit/{tag}.json';r=json.loads(p.read_text());assert r['status']=='completed'
  for name,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
  parents[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();v={x['arm']:x for x in r['rows']};rows.append(dict(seed=seed,gain_vs_frozen=v['frozen']['nll']-v['online']['nll'],gain_vs_zero=v['zero']['nll']-v['online']['nll'],gain_vs_bptt=v['bptt']['nll']-v['online']['nll'],online_whole_s=r['acquisition']['wall_s']+v['online']['wall_s'],bptt_whole_s=r['acquisition']['wall_s']+v['bptt']['wall_s']))
 consistent=all(x['gain_vs_frozen']>0 for x in rows);decision='Consistent adaptive intervention utility versus frozen policy; compare zero and complete cost separately.' if consistent else 'Adaptive intervention effect sign-inconsistent; retire scalar policy without further tuning.'
 result=dict(status='completed',battle='B10',tag=a.tag,rows=rows,decision=decision,mean_gain_vs_frozen=sum(x['gain_vs_frozen'] for x in rows)/3,mean_gain_vs_zero=sum(x['gain_vs_zero'] for x in rows)/3,parent_sha256=parents,scope='Three fixed synthetic conditional-change seeds; Gaussian innovations unpaired across arms. Shared acquisition charged. No real-market, changing-bias supremacy or clockless hardware claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
