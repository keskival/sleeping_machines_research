"""Fixed B10 functional-credit confirmation from completed source-bound evidence."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();rows=[];parents={}
 for seed in [187,188,189]:
  tag='curie_market_functional_future_20261012_v1_fit' if seed==187 else f'curie_market_functional_future_20261012_v1_s{seed}_fit'
  p=ROOT/f'experiments/results/credit/{tag}.json';r=json.loads(p.read_text());assert r['status']=='completed'
  for name,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
  parents[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();v={x['arm']:x for x in r['rows']};rows.append(dict(seed=seed,gain_vs_frozen=v['frozen']['nll']-v['online']['nll'],gain_vs_zero=v['zero']['nll']-v['online']['nll'],gain_vs_bptt=v['bptt']['nll']-v['online']['nll'],online_whole_s=r['acquisition']['wall_s']+v['online']['wall_s'],bptt_shared_whole_s=r['acquisition']['wall_s']+v['bptt']['wall_s']))
 consistent=all(x['gain_vs_zero']>0 and x['gain_vs_bptt']>0 for x in rows)
 result=dict(status='completed',battle='B10',tag=a.tag,rows=rows,consistent_quality=consistent,mean_gain_vs_frozen=sum(x['gain_vs_frozen'] for x in rows)/3,mean_gain_vs_zero=sum(x['gain_vs_zero'] for x in rows)/3,mean_gain_vs_bptt=sum(x['gain_vs_bptt'] for x in rows)/3,decision='Consistent development quality improvement; complete cost still decides economy.' if consistent else 'First-seed development quality win fails fixed confirmation; stop short-fragment tuning. Longer-data whole-learning test is separately registered.',parent_sha256=parents,scope='Three independent actor/critic starts on identical chronological Aug1 TRAIN fragment, not independent market episodes or sealed benchmark. Shared acquisition controls credit utility, not a baseline from original initialization.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
