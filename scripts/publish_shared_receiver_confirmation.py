"""B1/R1 source-bound shared-prefix confirmation and current report."""
import fcntl,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAGS=['curie_shared_receiver_fit_20261010T2300Z_pilot',*[f'curie_shared_receiver_confirm_s{s}_20261010T2320Z_pilot' for s in [171,172]]]
def main():
    rows=[];paths=[]
    for seed,tag in zip([170,171,172],TAGS):
        p=ROOT/'experiments/results/credit'/f'{tag}.json';d=json.loads(p.read_text());assert d['status']=='completed'
        for name,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        assert hashlib.sha256((ROOT/d['checkpoint']).read_bytes()).hexdigest()==d['checkpoint_sha256']
        ref,new=d['rows'];rows.append(dict(seed=seed,reference=ref,shared=new,gain_retained=new['gain_vs_fixed']/ref['gain_vs_fixed'],query_work_ratio=new['query_supported_flops']/ref['query_supported_flops']));paths.extend([str(p.relative_to(ROOT)),d['checkpoint']])
    out=ROOT/'experiments/results/credit/curie_shared_receiver_confirm_20261010T2320Z_decision.json';assert not out.exists()
    result=dict(status='completed',battle='B1/R1',rows=rows,confirmation_gate=all(r['shared']['gain_vs_fixed']>0 and r['shared']['unseen_inactive_relative_mse']<1 and r['gain_retained']>=.9 and r['query_work_ratio']<1 for r in rows),gate_scope='Seed170 selected pilot; seeds171/172 confirm positive utility, inactive error below zero, at least90% point-gain retention and reduced supported query work. This is a synthetic frozen-actor development gate, not equal-quality dominance or matched whole-fit work.',scope='Same per-seed actor/labels/steps/optimizer and nearly matched parameters, different critic initialization. Native depth4 optional key writes; previous and new teacher/fitting work charged. Query comparison excludes common descriptors. No online actor fitting, deployment wall-speed, full learning-efficiency, asynchronous hardware or TEST claim.')
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main';out.write_text(json.dumps(result,indent=2)+'\n');paths.append(str(out.relative_to(ROOT)))
        h=ROOT/'experiments/HANDOFF.md';text=re.sub(r'\*\*Curie shared-prefix pilot completed\.\*\*.*?\n\n','',h.read_text(),count=1,flags=re.S);h.write_text(f"**Curie shared-prefix confirmation completed.** Gate{result['confirmation_gate']}; retained gains{[round(r['gain_retained'],3) for r in rows]}, query work ratios{[round(r['query_work_ratio'],3) for r in rows]}. Result credit/curie_shared_receiver_confirm_20261010T2320Z_decision.json. Frozen synthetic development gate; no whole-fit/TEST claim. Next: selected-route error analysis and task-valued acquisition before causal changing-weight integration.\n\n"+text);paths.append('experiments/HANDOFF.md')
        subprocess.run(['git','add','-f',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish shared prefix receiver confirmation and work gate',*paths],cwd=ROOT,check=True)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/build_curie_episode_report.py','--tag','curie_shared_receiver_confirmation_report_20261010T2330Z'],cwd=ROOT,check=True)
if __name__=='__main__':main()
