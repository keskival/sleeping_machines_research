"""Await both admitted native fits, queue frozen decision, publish its evidence.

This controller launches no trainer. All native evaluation goes through the
existing six-slot coordinator/run_safe. It stops after this comparison.
"""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_paired_decision_20261015_v1'
RESULT=ROOT/f'experiments/results/hierarchy/{TAG}.json'
STATE=ROOT/'.git/g_hier_pair_comparison_20261015_v1.state.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**fields):
    temp=STATE.with_suffix('.writing');temp.write_text(json.dumps(fields,indent=2)+'\n');temp.replace(STATE)

def submit(paths):
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    if marker.exists():return
    source=set()
    for path in paths:
        fitted=json.loads(path.read_text());source.update(fitted['source_sha256'])
        source.update([str(path.relative_to(ROOT)),fitted['final_weights'],fitted['feature_packet']['path']])
    source.update(['experiments/hierarchy/paired_composition_decision.py','tests/test_observed_composition_groups.py','experiments/theory/NATIVE_EMISSION_CHANNEL_AUDIT.md',str(Path(__file__).resolve().relative_to(ROOT))])
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=True)
    q.write_text('# G-HIER: frozen paired comparison and conditional value/timing audit;no fitting or TEST.\n'+TAG+' experiments/hierarchy/paired_composition_decision.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={name:sha(ROOT/name) for name in sorted(source)},result=str(RESULT.relative_to(ROOT)),requires=[dict(result=str(p.relative_to(ROOT)),stage='paired_composition_learning') for p in paths],timeout_s=900,rss_kb=3000000,vms_kb=12000000)
    packet=dict(host='ip-172-31-47-132',battle='G-HIER',purpose='Reuse both native fits/caches;score fixed TRAIN heads and conditional value/timing interventions. No new fitting/TEST scoring.',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]})
    temp=marker.with_suffix('.writing');temp.write_text(json.dumps(packet,indent=2)+'\n');temp.replace(marker)


def replace_paragraph(text,prefix,new):
    start=text.find(prefix)
    if start<0:return text+'\n\n'+new+'\n'
    end=text.find('\n\n',start)
    if end<0:end=len(text)
    return text[:start]+new+text[end:]


def publish(d):
    metrics='; '.join(f"{arm}: NLL{r['nll']:.6f}, accuracy{100*r['accuracy']:.2f}%" for arm,r in d['rows'].items())
    short=f"**G-HIER paired native comparison completed,15October:** {metrics}. {d['decision']} One native seed,TRAIN-only affine heads,reused synthetic DEV;TEST not model-scored. [Findings](hierarchy/CURRENT_FINDINGS.md)."
    paragraph=f"**Paired role-memory result.** {metrics}. {d['decision']} Scope: one native seed,64paired updates/128scene presentations per arm,TRAIN-only ridge.01 affine head,64reused synthetic DEV scenes and observed-geometry clustered uncertainty;TEST not model-scored. Equal presentations do not mean equal parameters/arithmetic. Source: `experiments/results/hierarchy/{TAG}.json`."
    rows=[]
    for arm,r in d['rows'].items():
        work=r['work_training'];inference=r['work_dev']
        rows.append(f"| {arm} | {r['nll']:.6f} | {100*r['accuracy']:.2f}% | {r['parameters']} | {work['candidate_scores']/128:.1f} | {work.get('role_projection_madds',0)/128:.1f} | {inference['candidate_scores']/64:.1f} |")
    section='''### Paired native role-memory composition

'''+paragraph+'''

| Native arm | DEV NLL | DEV accuracy | Parameters | Native candidate scores/training scene | Role forward MAdds/training scene | Native inference candidates/scene |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+'''

Both arms use128presentations/4096raw event presentations. Full fitting/inference FLOPs, special-function work and traffic remain unaudited; candidate scores and role MAdds are separate partial ledgers, not complete arithmetic. Native losing-value work, timer scores, role products/transports and logical state are charged in the bound result files. CPU process totals include fitting/scoring; no hardware latency/energy claim.
'''
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (ROOT/'.git/MERGE_HEAD').exists():raise RuntimeError('publication waits for active merge resolution')
        for name in ['experiments/HANDOFF.md','experiments/PRODUCT_ORDERS.md']:
            path=ROOT/name;text=path.read_text()
            if text.startswith('**G-HIER'):
                end=text.find('\n\n');text=short+text[end:]
            else:text=short+'\n\n'+text
            path.write_text(text)
        path=ROOT/'experiments/hierarchy/CURRENT_FINDINGS.md';path.write_text(replace_paragraph(path.read_text(),'**Paired role-memory result.**',paragraph))
        path=ROOT/'report/I_SCIENCE.md';text=path.read_text();at=text.index('\n## 5.',text.index('### Native representational hierarchy'))
        if '**Paired role-memory result.**' in text:text=replace_paragraph(text,'**Paired role-memory result.**',paragraph)
        else:text=text[:at]+'\n'+paragraph+'\n'+text[at:]
        path.write_text(text)
        path=ROOT/'REPORT.md';text=path.read_text();title='### Paired native role-memory composition'
        at=text.find(title)
        if at>=0:
            end=text.find('\n##',at+len(title));end=len(text) if end<0 else end;text=text[:at]+section+text[end:]
        else:
            at=text.index('## Appendix. G-HIER deep representational composition implementation');end=text.find('\n## ',at+4);end=len(text) if end<0 else end;text=text[:end]+'\n'+section+text[end:]
        path.write_text(text)
        log=ROOT/'.git/g_hier_pair_comparison_20261015_v1.split.log'
        with log.open('w') as output:subprocess.run([str(ROOT/'.venv-docker/bin/python'),'report/split_report.py'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
        files=['experiments/HANDOFF.md','experiments/PRODUCT_ORDERS.md','experiments/hierarchy/CURRENT_FINDINGS.md','REPORT.md']+['report/'+n for n in ['I_SCIENCE.md','I_SCIENCE.pdf','II_METHODS.md','II_METHODS.pdf','III_RECORD.md','III_RECORD.pdf']]
        subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
        subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Publish completed native paired composition comparison','--',*files],cwd=ROOT,check=True)


def main():
    paths=[ROOT/f'experiments/results/hierarchy/aws_g_hier_paired_{arm}_20261015_v1.json' for arm in ['frame','relational']]
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        if all(p.exists() and json.loads(p.read_text()).get('status')=='completed' for p in paths):
            submit(paths);status(stage='decision_admitted',result=str(RESULT.relative_to(ROOT)))
            break
        status(stage='waiting_for_both_native_fits');time.sleep(10)
    else:raise RuntimeError('bounded native fit wait expired;check resumable checkpoints')
    while time.monotonic()<deadline:
        if RESULT.exists() and (d:=json.loads(RESULT.read_text())).get('status')=='completed':
            publish(d);status(stage='completed',result=str(RESULT.relative_to(ROOT)),decision=d['decision']);return
        time.sleep(10)
    raise RuntimeError('bounded decision wait expired;no further work admitted')

if __name__=='__main__':
    try:main()
    except Exception as error:status(stage='stopped',error=repr(error));raise
