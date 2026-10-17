"""Source-bound retry of the ordered writer-input resource gate."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',required=True);a=parser.parse_args();assert a.version.startswith('v') and a.version[1:].isdigit()
    original=json.loads((PLAN/'addenda/000_aws_r1_writer_input_resource_20261017_v1.json').read_text())
    purpose=dict(original['purpose']);purpose['decision']='Full writer-input graph exceeded the3MKiB envelope in v1. Does exact non-reentrant checkpoint recomputation retain its VJP while fitting the same16-update integrated workload? No reduced model/history or long fit.'
    tag=f'aws_r1_writer_input_resource_20261017_{a.version}';q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
    text='# '+json.dumps(purpose)+'\n'+f'{tag} experiments/credit/rmr_writer_resource_gate.py --tag {tag}\n'
    if q.exists():assert q.read_text()==text
    else:q.write_text(text)
    names=list(original['slots']['3'][0]['source_sha256'])
    gate='experiments/results/diagnostics/aws_r1_review_repairs_20261017_v6.json';evidence=json.loads((ROOT/gate).read_text());assert evidence['gate']=='PASS'
    writer='experiments/r1_rmr_writer_input_credit.py';assert evidence['source_sha256'][writer]==sha(ROOT/writer)
    job=dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in names},result=f'experiments/results/diagnostics/{tag}.json',requires=[dict(result=gate,stage='review repair integrated numerical gate',source_sha256={writer:sha(ROOT/writer)})],timeout_s=7200,rss_kb=3000000,vms_kb=12000000)
    path=PLAN/f'addenda/000_aws_r1_writer_input_resource_20261017_{a.version}.json';packet=dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':[job]})
    if path.exists():assert json.loads(path.read_text())==packet
    else:
        tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
    print(path)
if __name__=='__main__':main()
