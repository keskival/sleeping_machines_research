"""Bounded, source-bound continuation; all fitting remains in guarded slot2."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
DEADLINE = time.monotonic()+3*3600


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_result(path):
    while not path.exists():
        if time.monotonic()>DEADLINE: raise TimeoutError(str(path))
        time.sleep(15)
    record=json.loads(path.read_text())
    assert record['status']=='completed'
    return record


def admit(tag, script, arguments, result, sources, requires, training):
    queue=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt'
    queue.parent.mkdir(exist_ok=False)
    queue.write_text('# R1: exact cached replay; decision retain learning and reduce complete training time at two acquired-data sizes\n'+tag+' '+script+' '+arguments+'\n')
    job=dict(name=tag,queue=str(queue.relative_to(ROOT)),queue_sha256=sha(queue),kind='r1_job',
        source_sha256=sources,result=result,requires=requires,timeout_s=1800 if training else 120,
        rss_kb=1500000 if training else 500000,vms_kb=8000000 if training else 4000000)
    packet=dict(host='ip-172-31-47-132',battle='R1',purpose='Exact replay quality/work confirmation',
        parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'2':[job]})
    addendum=PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s2.json')
    addendum.write_text(json.dumps(packet,indent=2)+'\n')
    print('admitted '+tag,flush=True)


def main():
    gate=wait_result(ROOT/'experiments/results/diagnostics/aws_window_replay_fit_gate_20261012_v1.json')
    assert gate['gate']=='PASS',gate
    for exposure,seed in [('512k',6),('128k',7),('512k',7),('128k',8),('512k',8)]:
        reference=(f'aws_integrated_common_prior_512k_fresh_20261012_s{seed}_v1' if exposure=='512k'
            else f'aws_integrated_common_prior_fresh_20261012_s{seed}_v1')
        saved=json.loads((ROOT/f'experiments/results/token_language/{reference}.json').read_text())
        assert saved['status']=='completed'
        for source,digest in gate['source_sha256'].items(): assert sha(ROOT/source)==digest,source
        tag=f'aws_integrated_window_replay_{exposure}_20261012_s{seed}_v1'
        text=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
        arguments=text.split('experiments/common_prior_token_language.py ',1)[1].replace(reference,tag)
        sources=dict(saved['source_sha256'])
        sources.pop('experiments/common_prior_token_language.py')
        source='experiments/window_replay_token_language.py';sources[source]=gate['source_sha256'][source]
        result=f'experiments/results/token_language/{tag}.json'
        admit(tag,source,arguments,result,sources,[dict(result='experiments/results/diagnostics/aws_window_replay_fit_gate_20261012_v1.json')],True)
        wait_result(ROOT/result)
        check=f'aws_window_replay_{exposure}_gate_20261012_s{seed}_v1'
        sources.update(saved['source_sha256'])
        source='experiments/analyze_window_replay.py';sources[source]=gate['source_sha256'][source]
        result=f'experiments/results/diagnostics/{check}.json'
        admit(check,source,f'--tag {check} --reference {reference} --candidate {tag}',result,sources,
            [dict(result=f'experiments/results/token_language/{tag}.json')],False)
        completed=wait_result(ROOT/result)
        assert completed['gate']=='PASS',completed
        print('passed '+check,flush=True)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_window_replay.py')],cwd=ROOT,check=True)


if __name__=='__main__': main()
