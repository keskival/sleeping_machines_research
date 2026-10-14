"""R1 paired long-history DEV after the integrated cross-host language fit."""
import json,time
import continue_window_replay as queue
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+14*86400
def main():
 run='aws_local_clip_receiver_d8_8m_20261014_s6_v1'
 r=queue.wait_result(ROOT/f'experiments/results/token_language/{run}.json')
 sources=dict(r['source_sha256'])
 for name in ('experiments/check_deep_language_quality.py','experiments/public_token_scoring.py'):sources[name]=queue.sha(ROOT/name)
 tag='aws_local_clip_receiver_d8_8m_quality_20261014_s6_v1';script='experiments/check_deep_language_quality.py'
 p=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';p.parent.mkdir(exist_ok=False)
 p.write_text('# R1: integrated language quality/work; decision retain acquired deep member versus savedD2 on identical8192DEV/1024history\n'+tag+' '+script+' --tag '+tag+' --run '+run+'\n')
 job=dict(name=tag,queue=str(p.relative_to(ROOT)),queue_sha256=queue.sha(p),kind='r1_job',source_sha256=sources,result=f'experiments/results/diagnostics/{tag}.json',requires=[],timeout_s=1800,rss_kb=1500000,vms_kb=8000000)
 packet=dict(host='ip-172-31-47-132',battle='R1',purpose='Paired quality/fullwork after integrated sparse native language acquisition',parent_manifest_sha256=queue.sha(queue.PLAN/'manifest.json'),slots={'1':[job]})
 p=queue.PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s1.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(p)
 result=queue.wait_result(ROOT/job['result']);print(json.dumps(result['rows']),flush=True)
if __name__=='__main__':main()
