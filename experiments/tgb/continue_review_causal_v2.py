"""AWS non-fitting controller: measured B5 pilot -> one full-validation epoch.

No TEST, no direct training process, no concurrent slot or source substitution.
"""
import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COORD=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'


def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()


def advance(packet_path):
    for p in (COORD/'addenda').glob('*.json'):
        r=json.loads(p.read_text())
        if r.get('b5_causal_pilot_packet')==packet_path:
            return dict(state='full_development_admitted',packet=str(p.relative_to(ROOT)),result=r['slots']['3'][0]['result'])
    packet=json.loads((ROOT/packet_path).read_text());jobs=packet['slots']['3'];pilot=jobs[-1]
    result_path=ROOT/pilot['result']
    if not result_path.exists():
        status=json.loads((COORD/'worker_recovery.status.json').read_text())
        names={j['name'] for j in jobs}
        failures=[r for k in ('errors','blocked') for r in status.get(k,[]) if r.get('job') in names]
        if failures:return dict(state='pipeline_failed',failures=failures,next='Repair the failed step under fresh sources/tags; no full fit')
        return dict(state='awaiting_pilot',result=pilot['result'])
    r=json.loads(result_path.read_text());assert r['status']=='completed'
    for p,h in r['source_sha256'].items():assert sha(p)==h,'Pinned pilot source changed'
    assert r['dataset']=='tgbl-review' and r['args']['max_train']==50000 and r['args']['max_val']==5000
    history=r['history'];assert len(history)==2
    if not all(math.isfinite(v[k]) for v in history for k in ('train_loss','val_mrr','train_wall_s','val_wall_s')):
        return dict(state='development_required',reason='Nonfinite pilot; no full fit')
    # Exact dataset sizes from the verified release in B5_TGB.md, not a score
    # extrapolation. A 1.75 timing margin screens the first full epoch.
    projected=max(v['train_wall_s']/50000*3413837+v['val_wall_s']/5000*730784 for v in history)
    reserve=1.75*projected
    if reserve>21600 or r['max_rss_kb']*1.5>6000000:
        return dict(state='engineering_required',projected_epoch_s=projected,reservation_s=reserve,
                    pilot_max_rss_kb=r['max_rss_kb'],next='Repair throughput/memory before full data; no automatic long fit')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());name='aws_b5_review_causal2_full_dev_'+stamp
    q=Path('experiments/queue/b5')/name/(name+'.txt')
    sources=pilot['source_sha256'];source='experiments/tgb/race_link_review_v2.py'
    out=COORD/'addenda'/f'zzzzzzzzzzzzzzzzzzzzzzzzzzzb5_{stamp}_s3.json'
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        qabs=ROOT/q;qabs.parent.mkdir(parents=True,exist_ok=False)
        qabs.write_text(f'# B5: measured causal pilot -> one full DEV epoch; no TEST\n{name} {source} --tag {name} --epochs 1 --dim 16 --neg 20 --lr .003 --weight-decay .0001\n')
        requires=[dict(result=j['result'],source_sha256=j['source_sha256']) for j in jobs]
        requires[-1]['rss_margin']=1.5
        job=dict(name=name,queue=str(q),queue_sha256=sha(q),kind='b5_job',source_sha256=sources,
                 result=f'experiments/results/tgb/{name}.json',timeout_s=21600,rss_kb=6000000,vms_kb=12000000,requires=requires)
        payload=dict(host='ip-172-31-47-132',battle='B5',purpose='One full per-query-causal validation epoch after passed contracts/smoke/pilot and measured resource screen; no TEST. Compare full validation with GraphMixer 0.428 and diagnose query classes.',
                     parent_manifest_sha256=sha(str((COORD/'manifest.json').relative_to(ROOT))),b5_causal_pilot_packet=packet_path,
                     pilot_projection=dict(full_epoch_s=projected,timing_margin=1.75,pilot_max_rss_kb=r['max_rss_kb']),slots={'3':[job]})
        assert not out.exists();out.write_text(json.dumps(payload,indent=2)+'\n')
        paths=[str(q),str(out.relative_to(ROOT))]
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Admit measured causal review full-development epoch','--',*paths],cwd=ROOT,check=True)
    return dict(state='full_development_admitted',packet=str(out.relative_to(ROOT)),result=job['result'],projected_epoch_s=projected,
                scope='Timing projection only; prefix MRR is not compared with full-validation references')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--packet',required=True);ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    directory=ROOT/'.git/aws-b5-preview'/Path(a.packet).stem;directory.mkdir(parents=True,exist_ok=True)
    state=directory/'state.json'
    while True:
        result=advance(a.packet);tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(state)
        print(json.dumps(result),flush=True)
        if not a.watch or result['state']!='awaiting_pilot':break
        time.sleep(30)
