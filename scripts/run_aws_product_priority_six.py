"""Six guarded AWS slots, six physical cores and eight GiB host reserve with per-job source binding and preserved recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    path = Path(args.manifest)
    plan = json.loads(path.read_text())
    assert os.uname().nodename == plan['host']
    status = path.parent/'worker_recovery.aws6.status.json'
    history=json.loads((path.parent/'worker_recovery.status.json').read_text())
    retired=set(history.get('completed',[]))|{x['job'] for x in history.get('errors',[])}|{x['job'] for x in history.get('blocked',[])}
    for slot in range(1,7):plan['slots'].setdefault(str(slot),[])
    assert not status.exists(), 'Inspect expanded coordinator state before restarting'
    assert plan['host']=='ip-172-31-47-132'
    own_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runner=Path('experiments/queue/run_safe_aws6.sh')
    runner_sha=hashlib.sha256(runner.read_bytes()).hexdigest()
    mutex = threading.RLock()
    failed = threading.Event()
    state = dict(status='waiting_host_lock', active={}, completed=[],
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

    def save():
        with mutex:
            tmp = status.with_suffix('.tmp')
            tmp.write_text(json.dumps(state, indent=2)+'\n')
            tmp.replace(status)

    def frozen(job):
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==own_sha
        assert hashlib.sha256(runner.read_bytes()).hexdigest()==runner_sha
        for name, digest in {**plan['coordinator_sha256'], **job['source_sha256']}.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256(Path(job['queue']).read_bytes()).hexdigest() == job['queue_sha256']

    def git(*command):
        return subprocess.check_output(['git', *command], text=True).strip()

    def publish(files, message):
        with open('/tmp/aws-language-publication.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            git('add', '-f', '--', *files)
            git('commit', '--only', '-m', message, '--', *files)
            for _ in range(6):
                git('pull', '--rebase')
                if subprocess.run(['git', 'push', 'origin', 'main']).returncode == 0:
                    return
            raise RuntimeError('Publication retries exhausted')

    save()
    with open('/tmp/experiments-runner.lock', 'a') as reservation:
        fcntl.flock(reservation, fcntl.LOCK_EX)
        state['status'] = 'running'
        save()

        def ready(job):
            for required in job.get('requires', []):
                source = Path(required['result'])
                if not source.exists():
                    raise ValueError('Missing predecessor: '+str(source))
                record = json.loads(source.read_text())
                if record.get('status') != 'completed':
                    raise ValueError('Predecessor did not complete: '+str(source))
                for name, digest in required.get('source_sha256', {}).items():
                    if record.get('source_sha256', {}).get(name) != digest:
                        raise ValueError('Predecessor source mismatch: '+str(source))
                for field in ('manifest_sha256', 'stage'):
                    if field in required and record.get(field) != required[field]:
                        raise ValueError('Predecessor binding mismatch: '+str(source))
                if required.get('rss_margin'):
                    if record['max_rss_kb'] * required['rss_margin'] > job['rss_kb']:
                        raise ValueError('Measured smoke needs a larger reservation: '+str(source))

        def execute(slot, job):
            frozen(job)
            if job.get('rss_kb',3000000)>3000000:
                raise ValueError('Six-slot envelope permits <=3000000KiB per job; large jobs need a lower-concurrency reservation')
            env = dict(os.environ, AWS_GYM_SLOT=str(slot),
                       AWS_GYM_HOST_LOCK_FD=str(reservation.fileno()),
                       MEM_CAP_KB=str(job.get('vms_kb', 24000000)), MEM_CAP_RSS_KB=str(job.get('rss_kb', 3000000)),
                       MIN_AVAIL_MB='8192', JOB_TIMEOUT_S=str(job['timeout_s']),
                       TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
            with mutex:
                state['active'][str(slot)] = dict(job=job)
                save()
            subprocess.run(['taskset','-c',str(slot-1),'bash',str(runner),job['queue']],
                           env=env, pass_fds=(reservation.fileno(),), check=True)
            frozen(job)
            files = [job['queue'], 'experiments/queue/logs/'+job['name']+'.log',
                     str(Path(job['queue']).parent/('runner_'+Path(job['queue']).stem+'.out'))]
            if job.get('result'):
                result = json.loads(Path(job['result']).read_text())
                assert result['status'] == 'completed'
                if job['kind'] == 'dense_reference':
                    assert result['source_sha256'] == job['source_sha256'][result['script']]
                    files.extend(str(p) for p in Path(job['result']).parent.glob('*') if p.suffix in ('.json', '.pt'))
                else:
                    for name, digest in result['source_sha256'].items():
                        assert digest == job['source_sha256'].get(name,hashlib.sha256(Path(name).read_bytes()).hexdigest()), name
                if result.get('final_weights'):
                    files.append(result['final_weights'])
                if job['kind'] == 'mackey_glass':
                    assert result['primary_mode'] == 'mix8'
                    assert [r['repeat'] for r in result['repeats']] == list(range(job['first_repeat'], job['first_repeat']+10))
                elif job['kind'] == 'streaming':
                    files.append(str(Path(job['result']).with_suffix('.progress.pt')))
                files.append(job['result'])
            publish(files, 'Publish guarded benchmark continuation '+job['name'])
            with mutex:
                state['completed'].append(job['name'])
                state['active'].pop(str(slot), None)
                save()

        def worker(slot, jobs):
            pending = list(jobs)
            admitted = {j['name'] for j in jobs}
            loaded = set()
            while pending or plan.get('accept_addenda', False):
                if not pending:
                    for extra in sorted((path.parent/'addenda').glob('*.json')):
                        if str(extra) in loaded:
                            continue
                        packet = json.loads(extra.read_text())
                        assert packet['host'] == plan['host']
                        assert packet['parent_manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
                        for candidate in packet.get('slots', {}).get(str(slot), []):
                            assert candidate['name'] not in admitted
                            candidate = dict(candidate)
                            candidate['source_sha256'] = {**candidate['source_sha256'], str(extra): hashlib.sha256(extra.read_bytes()).hexdigest()}
                            pending.append(candidate)
                            admitted.add(candidate['name'])
                        loaded.add(str(extra))
                    if not pending:
                        with mutex:
                            state.setdefault('idle_slots', {})[str(slot)] = 'awaiting immutable addendum'
                            save()
                        time.sleep(30)
                        continue
                job = pending.pop(0)
                if job['name'] in retired or not Path(job['queue']).exists():
                    continue
                if job.get('result') and Path(job['result']).exists():
                    existing=json.loads(Path(job['result']).read_text())
                    if existing.get('status')=='completed':continue
                with mutex:
                    state.setdefault('idle_slots', {}).pop(str(slot), None)

                try:
                    ready(job)
                except Exception as error:
                    with mutex:
                        state.setdefault('blocked', []).append(dict(slot=slot, job=job['name'], error=str(error)))
                        save()
                    continue
                try:
                    execute(slot, job)
                except Exception as error:
                    # A failed diagnostic blocks its descendants through ready(),
                    # while unrelated arms and controls retain their admission.
                    with mutex:
                        state.setdefault('errors', []).append(dict(slot=slot, job=job['name'], error=str(error)))
                        state['active'].pop(str(slot), None)
                        save()
                    print('JOB FAILED '+job['name']+': '+str(error), flush=True)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(worker, int(slot), jobs) for slot, jobs in plan['slots'].items()]
            try:
                for future in futures:
                    future.result()
                state['status'] = 'needs_review' if state.get('errors') or state.get('blocked') else 'completed'
                save()
            except BaseException:
                state['status'] = 'needs_review'
                save()
                raise


if __name__ == '__main__':
    main()
