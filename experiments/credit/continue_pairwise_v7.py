"""Non-training controller: await pilot verdict; admit guarded confirmation only
if predeclared gates pass. All fitting stays in the existing run_safe scheduler.
Also serves as the scheduler's final confirmation analysis job.
"""
import argparse
import fcntl
import subprocess
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COORD=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'


def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()


def passed(record):
    row=next(r for r in record['rows'] if r['arm']=='pairwise_persistent')
    return row['dev_gap_to_dense']>=-.02 and row['credit_kl']<=.02


def summarize(paths):
    records=[json.loads((ROOT/p).read_text()) for p in paths]
    assert all(r['status']=='completed' for r in records)
    assert len({r['data_sha256'] for r in records})==1
    keys=('depth','K','K2','d','C','n_train','n_test','epochs','batch','data_seed','lr')
    assert len({tuple(r['args'][k] for k in keys) for r in records})==1
    assert len(records)==6
    rows=[]
    for seed in (0,1,2):
        dense=next(r for r in records if r['args']['seed']==seed and r['args']['arm']=='dense')
        learned=next(r for r in records if r['args']['seed']==seed and r['args']['arm']=='pairwise_persistent')
        rows.append(dict(seed=seed,dense_dev_ll=dense['final_dev_marginal_ll'],learned_dev_ll=learned['final_dev_marginal_ll'],
                         dev_gap=learned['final_dev_marginal_ll']-dense['final_dev_marginal_ll'],credit_kl=learned['final_credit_kl'],
                         dense_fit_linear_macs=dense['whole_fit_linear_macs'],learned_fit_linear_macs=learned['whole_fit_linear_macs'],
                         dense_macs_per_target=dense['macs_per_example'],learned_macs_per_target=learned['macs_per_example'],
                         dense_optimizer_visits=dense['optimizer_parameter_visits'],learned_optimizer_visits=learned['optimizer_parameter_visits'],
                         dense_wall_s=dense['wall_s'],learned_wall_s=learned['wall_s']))
    gaps=[r['dev_gap'] for r in rows]
    return dict(status='completed',evidence_level='three learner seeds, one fixed synthetic teacher, development only',
                rows=rows,mean_paired_dev_gap=statistics.mean(gaps),sd_paired_dev_gap=statistics.stdev(gaps),
                development_gate=all(r['dev_gap']>=-.02 and r['credit_kl']<=.02 for r in rows),
                next_decision='Integrate into a causal hidden-route model only if the development gate passes; otherwise diagnose coverage, approximation and chain tracking',
                scope='no public benchmark verdict; modeled linear work excludes optimizer/nonlinear FLOPs; shared-host wall time',
                inputs={p:sha(p) for p in paths})


def admit(packet_path):
    # Resume without duplicating a previously admitted confirmation.
    for path in sorted((COORD/'addenda').glob('*.json')):
        existing=json.loads(path.read_text())
        if existing.get('pilot_packet')==packet_path:
            return dict(state='confirmation_admitted',packet=str(path.relative_to(ROOT)),analysis_result=existing['slots']['1'][-1]['result'])
    packet=json.loads((ROOT/packet_path).read_text());jobs=packet['slots']['1']
    analysis=next(j for j in jobs if j['name'].endswith('_analysis'))
    result_path=ROOT/analysis['result']
    if not result_path.exists():
        status=COORD/'worker_recovery.status.json'
        if status.exists():
            live=json.loads(status.read_text()); names={j['name'] for j in jobs}
            names.update(Path(r['result']).stem for r in analysis.get('requires',[]))
            failures=[r for key in ('errors','blocked') for r in live.get(key,[]) if r.get('job') in names]
            if failures:return dict(state='pipeline_failed',failures=failures,next='Repair the specific failed contract or smoke under a fresh tag; no scaling')
        return dict(state='awaiting_pilots',result=analysis['result'])
    record=json.loads(result_path.read_text());assert record['status']=='completed'
    for p,h in record['source_sha256'].items():assert sha(p)==h,'analysis source changed'
    if not passed(record):
        return dict(state='development_required',decision=record['decision'],rows=record['rows'],
                    next='Read posterior-gap trajectories and persistent/reset contrast before selecting another variant; no automatic scaling')
    me=str(Path(__file__).resolve().relative_to(ROOT));stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    prefix='aws_pairwise7_confirm_'+stamp
    smoke=next(j for j in jobs if j['name'].endswith('_smoke'))
    learned0=next(j for j in jobs if j['name'].endswith('_pairwise_persistent'))
    contract=next(j for j in jobs if j['name'].endswith('_contract'))
    reference_path=next(r['result'] for r in analysis['requires'] if r['result'].endswith('_dense.json'))
    dense_source='experiments/credit/hindsight_race_v6.py'
    # No guessed or retuned settings. Replicate exactly the fixed pilot protocol.
    originals=[json.loads((ROOT/p).read_text()) for p in (reference_path,learned0['result'])]
    assert all(r['args']['epochs']==10 and r['args']['n_train']==20000 and r['args']['data_seed']==1234 for r in originals)
    for r in originals:
        for p,h in r['source_sha256'].items():assert sha(p)==h,'pinned pilot source changed'
    req=lambda j:dict(result=j['result'],source_sha256=j['source_sha256'])
    newjobs=[]
    for seed in (1,2):
        for arm in ('dense','pairwise_persistent'):
            name=f'{prefix}_{arm}_s{seed}';source=dense_source if arm=='dense' else 'experiments/credit/pairwise_race_v7.py'
            deps=[] if arm=='dense' else [dense_source]
            q=Path('experiments/queue/enabler')/name/(name+'.txt');(ROOT/q).parent.mkdir(parents=True,exist_ok=False)
            args=f'--arm {arm} --depth 2 --epochs 10 --seed {seed} --credit-hidden '+('64' if arm=='dense' else '8')
            (ROOT/q).write_text(f'# B1/R1 route-credit enabler: source-bound three-seed confirmation after DEV/credit gates\n{name} {source} --tag {name} {args}\n')
            smokereq=req(smoke);smokereq['rss_margin']=1.5
            newjobs.append(dict(name=name,queue=str(q),queue_sha256=sha(q),kind='enabler_job',source_sha256={p:sha(p) for p in (source,*deps)},
                                result=f'experiments/results/credit/{name}.json',timeout_s=1800,rss_kb=3000000,vms_kb=8000000,
                                requires=[req(contract),smokereq,req(analysis)]))
    all_results=[reference_path,learned0['result'],*[j['result'] for j in newjobs]]
    name=prefix+'_analysis';q=Path('experiments/queue/enabler')/name/(name+'.txt');(ROOT/q).parent.mkdir(parents=True,exist_ok=False)
    (ROOT/q).write_text(f'# B1/R1 route-credit enabler: three learner seeds, same teacher, final DEV and full learner work\n{name} {me} --tag {name} --summarize '+' '.join(all_results)+'\n')
    newjobs.append(dict(name=name,queue=str(q),queue_sha256=sha(q),kind='enabler_job',source_sha256={me:sha(me)},
                        result=f'experiments/results/credit/{name}.json',timeout_s=120,rss_kb=250000,vms_kb=8000000,
                        requires=[req(j) for j in newjobs]))
    payload=dict(pilot_packet=packet_path,host='ip-172-31-47-132',battle='ENABLER',purpose='B1/R1 pairwise credit: gated three-seed development confirmation; no benchmark test or architecture expansion',
                 parent_manifest_sha256=sha(str((COORD/'manifest.json').relative_to(ROOT))),slots={'1':newjobs})
    out=COORD/'addenda'/f'zzzzzzzzzzzzzzzzzzzzzzzzzzzenabler_{stamp}_s1.json';assert not out.exists()
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n')
    # Preserve the admission packet/queues on main, serialized with result publication.
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tmp.rename(out)
        paths=[str(out.relative_to(ROOT)),*[j['queue'] for j in newjobs]]
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','Admit gated pairwise-credit development confirmation','--',*paths],cwd=ROOT,check=True)
    return dict(state='confirmation_admitted',packet=str(out.relative_to(ROOT)),analysis_result=newjobs[-1]['result'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--packet');ap.add_argument('--watch',action='store_true');ap.add_argument('--tag');ap.add_argument('--summarize',nargs='+');a=ap.parse_args()
    if a.summarize:
        if not a.tag:ap.error('--tag required with --summarize')
        result=summarize(a.summarize);result.update(tag=a.tag,battle='B1/R1 route-credit enabler',source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(str(Path(__file__).resolve().relative_to(ROOT)))})
        with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result,indent=2))
    else:
        if not a.packet:ap.error('--packet required for controller')
        state_dir=ROOT/'.git/reciprocal-preview'/Path(a.packet).stem;state_dir.mkdir(parents=True,exist_ok=True)
        status_path=state_dir/'state.json'
        if status_path.exists() and json.loads(status_path.read_text())['state']!='awaiting_pilots':
            print(status_path.read_text());raise SystemExit(0)
        while True:
            state=admit(a.packet);tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status_path)
            print(json.dumps(state),flush=True)
            if state['state']!='awaiting_pilots' or not a.watch:break
            time.sleep(30)
