"""Recover admitted curie jobs sequentially through run_safe; source pins stay fixed."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with (ROOT / path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text())
    if os.uname().nodename != manifest['host']:
        raise ValueError('Host ownership mismatch')
    state_dir = ROOT / '.git/curie-recovery-preview' / manifest['tag']
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='checking', current_job=None, completed=[], started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))

    def save():
        temporary = state_dir / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(state_dir / 'state.json')

    def validate(job):
        record = json.loads((ROOT / job['result']).read_text())
        if record.get('status') != 'completed' or record.get('tag') != job['tag']:
            raise ValueError('Result incomplete or identity differs: ' + job['tag'])
        hashes = record.get('source_sha256')
        if isinstance(hashes, dict):
            if any(hashes.get(p) != h for p, h in job['result_sources'].items()):
                raise ValueError('Result sources differ: ' + job['tag'])
        elif hashes != job['result_sources'][job['driver']]:
            raise ValueError('Result driver differs: ' + job['tag'])
        return record

    try:
        for job in manifest['jobs']:
            for source, expected in job['sources'].items():
                if digest(source) != expected:
                    raise ValueError('Pinned source changed: ' + source)
            if digest(job['queue']) != job['queue_sha256']:
                raise ValueError('Pinned queue changed: ' + job['queue'])
            if (ROOT / job['result']).exists():
                validate(job); state['completed'].append(job['result']); save(); continue
            for required in job.get('requires', []):
                validate(next(j for j in manifest['jobs'] if j['tag'] == required))
            queue = job['queue']
            if job.get('recovery') and (ROOT / job['recovery']).exists():
                suffix = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
                resume_queue = (ROOT / queue).with_name(job['tag'] + '_resume_' + suffix + '.txt')
                lines = (ROOT / queue).read_text().splitlines()
                resume_queue.write_text('\n'.join(line if not line or line.startswith('#') else
                                                 line.replace(job['tag'] + ' ', job['tag'] + '_resume_' + suffix + ' ', 1) + ' --resume'
                                                 for line in lines) + '\n')
                queue = str(resume_queue.relative_to(ROOT))
            state.update(status='running', current_job=job['tag'], queue=queue); save()
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MEM_CAP_KB=str(job['vms_kb']), MEM_CAP_RSS_KB=str(job['rss_kb']),
                       MIN_AVAIL_MB=os.environ.get('MIN_AVAIL_MB', '10240'),
                       JOB_TIMEOUT_S=str(job['timeout_s']), WAIT='1')
            subprocess.run(['bash', 'experiments/queue/run_safe.sh', queue], cwd=ROOT, env=env, check=True)
            validate(job); state['completed'].append(job['result']); save()
        state.update(status='completed', current_job=None); save()
    except Exception as error:
        state.update(status='needs_review', error=str(error)); save(); raise


if __name__ == '__main__':
    main()
