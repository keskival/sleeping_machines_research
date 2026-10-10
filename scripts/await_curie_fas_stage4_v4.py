"""Curie B3: await selected reference artifacts, then guarded once-only scoring.

No training, remote host access or TEST array reads in this watcher. It commits
fresh source-bound admissions/one-job queues on main before run_safe execution.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/fas'))
from sealed_registry import atomic_json, sha

DATA = 'fas_v2_K2_drop0.02_delta0_20261005'
TOOLS = ('experiments/fas/score_sealed_v3.py','experiments/fas/sealed_registry.py',
         'experiments/fas/stage5_decision_v2.py','experiments/fas/stage5_statistics_v2.py',
         'experiments/fas/baselines.py','experiments/fas/payload_identity.py',
         'experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md')


def selected_records():
    records, missing = [], []
    for kind, seeds in (('native',(6,7,8)),('lstm',(0,1,2)),('transformer',(0,1,2))):
        for seed in seeds:
            if kind == 'native':
                tag=f'curie_b3_stage4_c10_s{seed}_20261009T0600Z'
                path=Path('experiments/results/fas')/(tag+'.json')
            else:
                stamp='20261006T1815Z' if seed == 0 else '20261009T1330Z'
                tag=f'aws_fas_v2_ref_{kind}_d128_lr0.003_s{seed}_{stamp}'
                path=Path('experiments/results/aws_20260929')/tag/(tag+'.json')
            if not (ROOT/path).exists(): missing.append(str(path));continue
            record=json.loads((ROOT/path).read_text())
            if record['status'] != 'completed': missing.append(str(path)+' (not completed)');continue
            checkpoint=Path(record['checkpoint']) if kind == 'native' else path.parent/record['selected_weights']['path']
            if not (ROOT/checkpoint).exists(): missing.append(str(checkpoint));continue
            records.append((kind,seed,tag,path,checkpoint,record))
    return records,missing


def prepare(tag, contract, preflight):
    reference_manifest=Path('experiments/results/fas/aws_fas_v2_data_manifest.json')
    if not (ROOT/reference_manifest).exists():return None,[str(reference_manifest)]
    records,missing=selected_records()
    if missing: return None,missing
    if len(records) != 9: raise ValueError('Nine selected native/reference fits required')
    for kind in ('native','lstm','transformer'):
        configs=[{k:v for k,v in r['args'].items() if k not in ('seed','tag')}
                 for family,_,_,_,_,r in records if family == kind]
        if len({json.dumps(c,sort_keys=True) for c in configs}) != 1:
            raise ValueError('Selected family confirmation configurations differ: '+kind)
    data_hash=sha(ROOT/'experiments/data/fas'/DATA/'manifest.json')
    sources={path:sha(ROOT/path) for path in TOOLS}
    directory=ROOT/'experiments/queue'/tag
    if directory.exists(): raise FileExistsError('Fresh scoring pipeline name required')
    directory.mkdir()
    jobs=[contract,preflight]; output_records={kind:[] for kind in ('native','lstm','transformer')}
    tracked=[]
    def admit(kind,seed,trained_tag,path,checkpoint,record):
        pins=dict(sources,**record['source_sha256'])
        admission=dict(kind=kind,data=DATA,data_manifest_sha256=data_hash,trained_tag=trained_tag,
                       source_sha256=pins,result=str(path),result_sha256=sha(ROOT/path),
                       reference_manifest=str(reference_manifest))
        if checkpoint is not None:
            admission.update(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(ROOT/checkpoint))
        else: admission['fit_runs']=2000
        from score_sealed_v3 import verify_admission
        verify_admission(admission,root=ROOT)
        for source,digest in pins.items():
            if sha(ROOT/source) != digest: raise ValueError('Selected source changed: '+source)
        name=f'{tag}_{kind}'+(f'_s{seed}' if seed is not None else '')
        admission_path=directory/(name+'_admission.json');atomic_json(admission_path,admission)
        queue=directory/(name+'.txt')
        queue.write_text('# B3 Stage 4, no fitting; decision: sealed confirmation of the selected configuration\n'+
                         f'{name} experiments/fas/score_sealed_v3.py --tag {name} --admission {admission_path.relative_to(ROOT)} --admission-sha256 {sha(admission_path)}\n')
        output=f'experiments/results/fas/{trained_tag}_TEST.json'
        job=dict(tag=name,driver='experiments/fas/score_sealed_v3.py',queue=str(queue.relative_to(ROOT)),
                 queue_sha256=sha(queue),sources=dict(pins,**{str(admission_path.relative_to(ROOT)):sha(admission_path),
                                                          str(path):sha(ROOT/path),str(reference_manifest):sha(ROOT/reference_manifest)}),
                 result_sources=pins,result=output,timeout_s=7200,rss_kb=3000000,vms_kb=8000000,
                 requires=[contract['tag'],preflight['tag']])
        jobs.append(job);tracked.extend([str(admission_path.relative_to(ROOT)),str(queue.relative_to(ROOT))])
        return output
    for kind,seed,trained_tag,path,checkpoint,record in records:
        output_records[kind].append(admit(kind,seed,trained_tag,path,checkpoint,record))
    classical_path=Path('experiments/results/fas/fas_v2_classical_val_20261006T2115Z.json')
    classical=json.loads((ROOT/classical_path).read_text())
    classical_output=admit('classical',None,tag+'_classical',classical_path,None,classical)
    decision_tag=tag+'_decision';queue=directory/(decision_tag+'.txt')
    arguments=' '.join('--'+kind+' '+' '.join(paths) for kind,paths in output_records.items())
    output=f'experiments/results/fas/{decision_tag}.json'
    queue.write_text('# B3 Stage 5; decision: apply the registered primary rule to completed ledger-bound scores\n'+
                     f'{decision_tag} experiments/fas/stage5_decision_v2.py --tag {decision_tag} {arguments} --classical {classical_output} --out {output}\n')
    jobs.append(dict(tag=decision_tag,driver='experiments/fas/stage5_decision_v2.py',queue=str(queue.relative_to(ROOT)),
                     queue_sha256=sha(queue),sources=sources,result_sources={path:sources[path] for path in (
                         'experiments/fas/stage5_decision_v2.py','experiments/fas/stage5_statistics_v2.py',
                         'experiments/fas/sealed_registry.py')},
                     result=output,timeout_s=1800,rss_kb=1500000,vms_kb=8000000,
                     requires=[job['tag'] for job in jobs]))
    tracked.append(str(queue.relative_to(ROOT)))
    manifest=directory/'manifest.json'
    atomic_json(manifest,dict(tag=tag,host='curie',battle='B3',training=False,jobs=jobs,
        ownership='Curie scores all selected checkpoints; AWS reference fitting remains AWS-owned',
        purpose='Frozen C10 native and selected LSTM/Transformer families, classical suite, then registered decision; no tuning'))
    tracked.append(str(manifest.relative_to(ROOT)))
    return (str(manifest.relative_to(ROOT)),tracked),[]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--contract-manifest',required=True)
    parser.add_argument('--native-preflight-manifest',required=True);args=parser.parse_args()
    if os.uname().nodename != 'curie': raise ValueError('Curie owns this scoring pipeline')
    contract=json.loads((ROOT/args.contract_manifest).read_text())['jobs'][0]
    preflight=json.loads((ROOT/args.native_preflight_manifest).read_text())['jobs'][0]
    expected={path:sha(ROOT/path) for path in TOOLS}
    state_dir=ROOT/'.git/fas-stage4-preview'/args.tag;state_dir.mkdir(parents=True,exist_ok=True)
    lock=(state_dir/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def save(status,**fields):
        atomic_json(state_dir/'state.json',dict(status=status,observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields))
    try:
        while True:
            if any(sha(ROOT/path) != digest for path,digest in expected.items()):
                raise ValueError('Scoring/protocol source changed while awaiting references')
            prepared,missing=prepare(args.tag,contract,preflight)
            if prepared is None:
                save('awaiting_selected_reference_artifacts',missing=missing);time.sleep(30);continue
            manifest,paths=prepared
            if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != 'main':
                raise ValueError('Main publication required before scoring')
            subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Admit source-bound Curie B3 sealed confirmation','--',*paths],cwd=ROOT,check=True)
            save('admitted_waiting_for_host_lock',manifest=manifest)
            subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_curie_recovery.py','--manifest',manifest],cwd=ROOT,check=True)
            save('completed',manifest=manifest);return
    except Exception as error:
        save('needs_review',error=str(error));raise


if __name__ == '__main__':main()
