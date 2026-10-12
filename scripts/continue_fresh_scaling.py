"""Bounded R1 2M pair -> source-bound decision -> selected-member 8M fit."""
import json
import subprocess
import time
import continue_window_replay as queue

ROOT = queue.ROOT
queue.DEADLINE = time.monotonic() + 8*3600


def admit(tag, driver, args, result, sources, requires, training):
    # Reuse the established single-slot admission, then set measured-size limits
    # before the coordinator can observe a job: use our own atomic addendum.
    path = ROOT/f'experiments/queue/r1/{tag}/{tag}.txt'
    path.parent.mkdir(exist_ok=False)
    path.write_text('# R1: fixed-prior fresh-acquisition scaling; decision select useful addressed capacity and larger learning frontier\n'
                    + tag+' '+driver+' '+args+'\n')
    job = dict(name=tag, queue=str(path.relative_to(ROOT)), queue_sha256=queue.sha(path), kind='r1_job',
        source_sha256=sources, result=result, requires=requires,
        timeout_s=(14400 if '8m' in tag else 5400) if training else 120,
        rss_kb=1500000 if training else 500000, vms_kb=8000000 if training else 4000000)
    packet = dict(host='ip-172-31-47-132', battle='R1', purpose='Fixed-prior fresh data/capacity learning frontier',
        parent_manifest_sha256=queue.sha(queue.PLAN/'manifest.json'), slots={'2':[job]})
    path = queue.PLAN/'addenda'/('zzzzzzzzzzzzzzzzzzzzzzzzzzzr1_'+str(time.time_ns())+'_s2.json')
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(packet, indent=2)+'\n'); tmp.replace(path)
    print('admitted '+tag, flush=True)


def fit(stage, pool):
    variant = 'window_replay' if pool == 4 else 'clock_capacity8'
    saved_tag = f'aws_integrated_{variant}_512k_20261012_s6_v1'
    saved = queue.wait_result(ROOT/f'experiments/results/token_language/{saved_tag}.json')
    sources = saved['source_sha256']
    for name, digest in sources.items(): assert queue.sha(ROOT/name) == digest, name
    driver = 'experiments/window_replay_token_language.py' if pool == 4 else 'experiments/capacity_clock_token_language.py'
    command = (ROOT/f'experiments/queue/r1/{saved_tag}/{saved_tag}.txt').read_text().splitlines()[1]
    tag = f'aws_integrated_fresh_scaling_{stage}_pool{pool}_20261012_s6_v1'
    args = command.split(driver+' ', 1)[1].replace(saved_tag, tag).replace('--steps 1024', '--steps '+('4096' if stage == '2m' else '16384'))
    result = f'experiments/results/token_language/{tag}.json'
    admit(tag, driver, args, result, sources, [dict(result=f'experiments/results/token_language/{saved_tag}.json')], True)
    return queue.wait_result(ROOT/result)


def analyze(stage, pool=4):
    tag = f'aws_fresh_scaling_{stage}_decision_20261012_v1'
    driver = 'experiments/analyze_fresh_scaling.py'
    sources = {driver:queue.sha(ROOT/driver)}
    for p in ((4,8) if stage == '2m' else (pool,)):
        for previous in ('512k', '2m', stage):
            variant = 'window_replay' if p == 4 else 'clock_capacity8'
            name = (f'aws_integrated_{variant}_512k_20261012_s6_v1' if previous == '512k'
                else f'aws_integrated_fresh_scaling_{previous}_pool{p}_20261012_s6_v1')
            path = ROOT/f'experiments/results/token_language/{name}.json'
            if path.exists(): sources.update(json.loads(path.read_text())['source_sha256'])
    result = f'experiments/results/diagnostics/{tag}.json'
    admit(tag, driver, f'--tag {tag} --stage {stage} --pool {pool}', result, sources, [], False)
    return queue.wait_result(ROOT/result)


def main():
    gate = queue.wait_result(ROOT/'experiments/results/diagnostics/aws_capacity_clock_contract_20261012_v2.json')
    assert all(r['feature_teacher_gradient_parameter_error'] == 0 for r in gate['rows'])
    for pool in (4,8): fit('2m', pool)
    d = analyze('2m')
    published = subprocess.run([str(ROOT/'.venv-docker/bin/python'), str(ROOT/'scripts/publish_fresh_scaling.py'), '--stage', '2m'], cwd=ROOT)
    if published.returncode: print('Publication needs recovery; preserve evidence and continue the authorized larger fit', flush=True)
    if not d['continue_larger']:
        print('8M stopped: final-learning/resource gate requires diagnosis', flush=True); return
    fit('8m', d['selected_pool']); analyze('8m', d['selected_pool'])
    subprocess.run([str(ROOT/'.venv-docker/bin/python'), str(ROOT/'scripts/publish_fresh_scaling.py'), '--stage', '8m'], cwd=ROOT, check=True)


if __name__ == '__main__': main()
