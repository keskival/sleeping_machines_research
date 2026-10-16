"""Await six repaired native fits,freeze five-split selection,TEST once,publish."""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_b4x_github_test_20261016_v3';OUT=ROOT/f'experiments/results/tpp_b4/{TAG}.json';STATE=ROOT/'.git/b4_github_memory_repair_20261016_v1.state.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**value):STATE.write_text(json.dumps(value,indent=2)+'\n')
def main():
    paths={v:[f'experiments/results/tpp_b4/aws_b4x_github_{v}_'+('20261016_v3' if i in [1,2,3] and v!='marks' else '20261015_v1')+f'_s{i}.json' for i in range(5)] for v in ['ptr','ptrlc','marks']}
    deadline=time.monotonic()+45000
    while time.monotonic()<deadline:
        if all((ROOT/p).exists() and json.loads((ROOT/p).read_text()).get('status')=='completed' for group in paths.values() for p in group):break
        status(stage='waiting_for_six_memory_repaired_fits');time.sleep(15)
    else:raise RuntimeError('bounded development wait expired;retain exact pass checkpoints')
    assert not (ROOT/'experiments/results/tpp_b4/aws_b4x_github_test_20261015_v1.json').exists()
    protocol=ROOT/f'experiments/queue/b4/{TAG}/selection.json';protocol.parent.mkdir(parents=True,exist_ok=True);protocol.write_text(json.dumps(dict(variants=paths,selection='minimum five-split DEV mean;old completed splits reused,memory-failed splits repaired;TEST once'),indent=2)+'\n')
    sources={'experiments/tpp/b4x_manifest_score.py','scripts/continue_b4_github_memory_repair.py',str(protocol.relative_to(ROOT))};requires=[]
    for group in paths.values():
        for name in group:
            r=json.loads((ROOT/name).read_text());sources.update(r['source_sha256']);sources.update([name,r['checkpoint']]);requires.append(dict(result=name))
    q=protocol.parent/(TAG+'.txt');q.write_text('# B4: frozen five-split development selection,TEST once,full history.\n'+TAG+' experiments/tpp/b4x_manifest_score.py --tag '+TAG+' --dataset github_filtered --manifest '+str(protocol.relative_to(ROOT))+'\n')
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    if not marker.exists():
        job=dict(name=TAG,kind='b4_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={p:sha(ROOT/p) for p in sorted(sources)},result=str(OUT.relative_to(ROOT)),requires=requires,timeout_s=2700,rss_kb=3000000,vms_kb=12000000)
        marker.write_text(json.dumps(dict(host='ip-172-31-47-132',battle='B4',purpose='Complete native Github developed attempt;validation selection and once-only TEST after memory repair.',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),indent=2)+'\n')
    status(stage='frozen_test_admitted')
    while time.monotonic()<deadline:
        if OUT.exists() and (r:=json.loads(OUT.read_text())).get('status')=='completed':break
        time.sleep(15)
    else:raise RuntimeError('bounded score wait expired;no new fitting admitted')
    verdict='WIN' if r['beats_composite_target'] else 'developed attempt below composite target'
    text=f"**B4 Github developed result,16October:** {verdict}: five-split recorded-data totalNLL{r['test_mean']['total']:.6f}±{r['test_total_se']:.6f}SE versus composite{r['composite_target']:.2f};best single published model{r['best_single_published']:.2f}. Selected{r['chosen_variant']} by all-five-split validation mean;TEST once. Native pointer histories retained;completed old splits reused and only memory-failed fits replaced. Source:`experiments/results/tpp_b4/{TAG}.json`. No further fitting automatically admitted."
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        names=['experiments/HANDOFF.md','experiments/PRODUCT_ORDERS.md','experiments/tpp/B4_DEVELOPED_CURRENT.md','REPORT.md','report/I_SCIENCE.md']
        for name in names:
            p=ROOT/name;s=p.read_text() if p.exists() else ''
            if name=='REPORT.md':
                title='## Appendix. B4 repaired learning and mark-memory development';at=s.index(title)+len(title);s=s[:at]+'\n\n'+text+s[at:]
            elif name=='report/I_SCIENCE.md':
                at=s.index('**B4 development:**');s=s[:at]+text+'\n\n'+s[at:]
            else:s=text+'\n\n'+s
            p.write_text(s)
        with (ROOT/'.git/b4_github_memory_repair_20261016.split.log').open('w') as log:subprocess.run([str(ROOT/'.venv-docker/bin/python'),'report/split_report.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        names+=['report/'+n for n in ['I_SCIENCE.pdf','II_METHODS.md','II_METHODS.pdf','III_RECORD.md','III_RECORD.pdf']]
        subprocess.run(['git','add','--',*names],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish completed B4 Github developed attempt','--',*names],cwd=ROOT,check=True)
    status(stage='completed',verdict=verdict);return
if __name__=='__main__':
    try:main()
    except Exception as error:status(stage='stopped',error=repr(error));raise
