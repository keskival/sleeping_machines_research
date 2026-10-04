"""Three guarded host slots with per-job source binding and preserved recovery."""
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

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    path = Path(args.manifest)
    plan = json.loads(path.read_text())
    assert os.uname().nodename == plan['host']
    status = path.parent/'worker_recovery.status.json'
    assert not status.exists()
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

        def worker(slot, jobs):
            try:
                for job in jobs:
                    if failed.is_set():
                        return
                    frozen(job)
                    env = dict(os.environ, AWS_GYM_SLOT=str(slot),
                               AWS_GYM_HOST_LOCK_FD=str(reservation.fileno()),
                               MEM_CAP_KB=str(job.get('vms_kb', 24000000)), MEM_CAP_RSS_KB=str(job.get('rss_kb', 4000000)),
                               MIN_AVAIL_MB='8192', JOB_TIMEOUT_S=str(job['timeout_s']),
                               TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
                    with mutex:
                        state['active'][str(slot)] = dict(job=job)
                        save()
                    subprocess.run(['bash', 'experiments/queue/run_safe.sh', job['queue']],
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
                                assert digest == job['source_sha256'][name], name
                        if job['kind'] == 'language_fit':
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
            except BaseException as error:
                failed.set()
                with mutex:
                    state.setdefault('errors', []).append(dict(slot=slot, error=str(error)))
                    save()
                raise

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, int(slot), jobs) for slot, jobs in plan['slots'].items()]
            try:
                for future in futures:
                    future.result()
                state['status'] = 'completed'
                save()
            except BaseException:
                state['status'] = 'needs_review'
                save()
                raise


if __name__ == '__main__':
    main()
