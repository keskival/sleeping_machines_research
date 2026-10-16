"""Finish admitted Stack Overflow development,repair only an unexecuted scorer."""
import hashlib,json,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
OLD='aws_b4x_so_test_20261015_v1';TAG='aws_b4x_so_test_20261016_v2';STATE=ROOT/'.git/b4_stack_scoring_20261016_v1.state.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**v):STATE.write_text(json.dumps(v,indent=2)+'\n')
def main():
    inputs=[f'experiments/results/tpp_b4/aws_b4x_so_{v}_20261015_v1_s{i}.json' for v in ['ptr','marks'] for i in range(5)];deadline=time.monotonic()+30000
    old=ROOT/f'experiments/results/tpp_b4/{OLD}.json';out=ROOT/f'experiments/results/tpp_b4/{TAG}.json';status(stage='awaiting_admitted_split4_fits')
    while time.monotonic()<deadline:
        if all((ROOT/p).exists() for p in inputs):break
        time.sleep(15)
    else:raise RuntimeError('bounded Stack Overflow development wait expired')
    while time.monotonic()<deadline:
        if old.exists():out=old;break
        state=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text())
        if any(r['job']==OLD for r in state.get('errors',[])):
            assert not (ROOT/f'experiments/queue/logs/{OLD}.log').exists(),'TEST execution may have begun;do not replay automatically'
            source={'experiments/tpp/b4x_score_stack_frozen_20261016.py','scripts/continue_b4_stack_scoring.py'}
            for name in inputs:
                r=json.loads((ROOT/name).read_text());assert r['status']=='completed';source.update(r['source_sha256']);source.update([name,r['checkpoint']])
            q=ROOT/f'experiments/queue/b4/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=True)
            q.write_text('# B4: restored unexecuted fixed five-split Stack Overflow scoring;no new fitting.\n'+TAG+' experiments/tpp/b4x_score_stack_frozen_20261016.py --tag '+TAG+' --dataset stack_overflow_filtered --variants aws_b4x_so_ptr_20261015_v1,aws_b4x_so_marks_20261015_v1\n')
            marker=PLAN/'addenda'/('000_'+TAG+'.json')
            if not marker.exists():
                job=dict(name=TAG,kind='b4_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(source)},result=str(out.relative_to(ROOT)),requires=[dict(result=p) for p in inputs],timeout_s=2700,rss_kb=3000000,vms_kb=12000000)
                marker.write_text(json.dumps(dict(host='ip-172-31-47-132',battle='B4',purpose='Restore existing strongest-variant DEV selection/once-only TEST after source preflight;no recipe change.',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'4':[job]}),indent=2)+'\n')
            status(stage='restored_frozen_scorer_admitted');break
        time.sleep(15)
    while time.monotonic()<deadline:
        if out.exists() and (r:=json.loads(out.read_text())).get('status')=='completed':break
        time.sleep(15)
    else:raise RuntimeError('bounded Stack Overflow test wait expired')
    verdict='WIN' if r['beats_composite_target'] else 'developed attempt below composite target'
    text=f"**B4 Stack Overflow developed result,16October:** {verdict};recorded-data five-split totalNLL{r['test_mean']['total']:.6f}±{r['test_total_se']:.6f}SE versus composite{r['composite_target']:.2f}. Variant selected by all-five-split DEV mean;TEST once. Source:`{out.relative_to(ROOT)}`. No further fitting automatically admitted."
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);files=['experiments/tpp/B4_DEVELOPED_CURRENT.md','experiments/HANDOFF.md','experiments/PRODUCT_ORDERS.md']
        for name in files:
            p=ROOT/name;p.write_text(text+'\n\n'+(p.read_text() if p.exists() else ''))
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish completed B4 Stack Overflow developed attempt','--',*files],cwd=ROOT,check=True)
    status(stage='completed',verdict=verdict)
if __name__=='__main__':
    try:main()
    except Exception as e:status(stage='stopped',error=repr(e));raise
