"""Read-only B1 continuation gate for completed whole-cohort learning."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--parent',required=True);a=ap.parse_args();path=ROOT/f'experiments/results/credit/{a.parent}.json';p=json.loads(path.read_text());assert p['status']=='completed' and p['gate']
 for name,digest in p['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 good=[r['depth'] for r in p['rows'] if r['online_gain_vs_adam']>0 and r['online_gain_vs_matched_heads']>0 and r['online_gain_vs_frozen']>0 and r['matched_quality_wall_ratio'] is not None and r['matched_quality_wall_ratio']>=1]
 out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();record=dict(status='completed',battle='B1',tag=a.tag,parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),quality_and_work_depths=good,admit_unchanged_seed_confirmation=len(good)>=2,scope='Registered two-depth joint quality/work gate; fresh seed confirmations admitted only after real native targets and wholeCPU acquisition-inclusive work pass. Development selection, not benchmark verdict.');out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
