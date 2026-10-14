"""Queue a unique guarded B10 budget-admission attempt on free slot4."""
import argparse,json,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/market'))
from b10_budget_admission import source_hashes,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();tag=a.tag
 plan=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z';q=ROOT/f'experiments/queue/aws_b10_matched_controls_20261014_v1/{tag}/{tag}.txt';q.parent.mkdir(exist_ok=False)
 q.write_text('# B10: freeze matched arithmetic and resource/resume contracts; decision admit nine full native/LSTM/Transformer references\n'+f'{tag} experiments/market/b10_budget_admission.py --tag {tag}\n')
 job=dict(name=tag,kind='b10_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=source_hashes(),result=f'experiments/results/market/{tag}.json',requires=[],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)
 p=plan/'addenda'/f'zzzzzzzzzzzzzzzzzzzzzzzzzzzzb10_{time.time_ns()}_s4.json';p.write_text(json.dumps(dict(host='ip-172-31-47-132',battle='B10',purpose='Founder-authorized full reference admission',parent_manifest_sha256=sha(plan/'manifest.json'),slots={'4':[job]}),indent=2)+'\n');print(p)
if __name__=='__main__':main()
