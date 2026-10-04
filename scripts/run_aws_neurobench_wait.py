"""Admit the assigned official battery after acquiring the normal host reservation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    path = Path(args.manifest)
    plan = json.loads(path.read_text())
    assert os.uname().nodename == plan['host']
    status = path.parent/'status.json'
    assert not status.exists()
    state = dict(status='waiting_normal_host_reservation', completed=[])

    def save():
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(status)

    def frozen():
        for name, digest in plan['sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name

    def git(*command):
        return subprocess.check_output(['git', *command], text=True).strip()

    save()
    try:
        with open('/tmp/experiments-runner.lock', 'a') as reservation:
            fcntl.flock(reservation, fcntl.LOCK_EX)
            frozen()
            assert git('branch', '--show-current') == 'main'
            for job in plan['jobs']:
                frozen()
                state.update(status='running', active=job['name'])
                save()
                env = dict(os.environ, AWS_GYM_SLOT='1',
                           AWS_GYM_HOST_LOCK_FD=str(reservation.fileno()),
                           MEM_CAP_KB='6000000', MEM_CAP_RSS_KB='2000000',
                           MIN_AVAIL_MB='8192', JOB_TIMEOUT_S=str(job['timeout_s']),
                           TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
                subprocess.run(['bash', 'experiments/queue/run_safe.sh', job['queue']],
                               env=env, pass_fds=(reservation.fileno(),), check=True)
                frozen()
                files = [job['queue'], 'experiments/queue/logs/'+job['name']+'.log',
                         str(Path(job['queue']).parent/('runner_'+Path(job['queue']).stem+'.out'))]
                if job.get('result'):
                    r = json.loads(Path(job['result']).read_text())
                    assert r['status'] == 'completed' and r['primary_mode'] == 'mix8'
                    assert [v['repeat'] for v in r['repeats']] == list(range(job['first_repeat'], job['first_repeat']+10))
                    for name, digest in r['source_sha256'].items():
                        assert digest == plan['sha256'][name], name
                    files.append(job['result'])
                with open('/tmp/aws-language-publication.lock', 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    git('add', '-f', '--', *files)
                    git('commit', '--only', '-m', 'Publish guarded NeuroBench '+job['name'], '--', *files)
                    git('pull', '--rebase')
                    git('push', 'origin', 'main')
                state['completed'].append(job['name'])
                save()
            state.update(status='completed', active=None)
            save()
    except BaseException as error:
        state.update(status='needs_review', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
