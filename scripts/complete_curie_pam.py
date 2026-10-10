"""B2 completion: retry exact checkpoint recovery only after memory-guard stops."""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def memory_stop(output):
    return bool(re.search(r'^\S+ STOP[^\n]*(?:MemAvailable=\d+MB below|cgroup headroom=)', output, re.MULTILINE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    if os.uname().nodename != 'curie':
        raise ValueError('Curie owns this controller')
    manifest = json.loads((ROOT / args.manifest).read_text())
    directory = ROOT / '.git/curie-pam-completion-preview' / manifest['tag']
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='waiting_for_memory', attempts=0)
    deadline = time.monotonic() + 86400

    def save():
        state['observed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temporary = directory / 'state.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(directory / 'state.json')

    while time.monotonic() < deadline and state['attempts'] < 12:
        available = next(int(line.split()[1]) // 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used = int(Path('/sys/fs/cgroup/memory.current').read_text())
        headroom = float('inf') if limit == 'max' else (int(limit) - used) / 1048576
        state.update(status='waiting_for_memory', available_mib=available, cgroup_headroom_mib=headroom if limit != 'max' else None)
        save()
        # Measured PAM RSS ~1.9 GiB; leave 2 GiB above the enforced 10 GiB host floor.
        if available < 12288 or headroom < 5120:
            time.sleep(30)
            continue
        state.update(status='running', attempts=state['attempts'] + 1)
        save()
        log = directory / f"attempt_{state['attempts']}.log"
        with log.open('w') as stream:
            outcome = subprocess.run([str(ROOT / '.venv-docker/bin/python'), 'scripts/run_curie_recovery.py',
                                      '--manifest', args.manifest], cwd=ROOT,
                                     env=dict(os.environ, MIN_AVAIL_MB='10240'), stdout=stream, stderr=subprocess.STDOUT)
        if outcome.returncode == 0:
            state['status'] = 'completed'
            save()
            return
        output = log.read_text()
        if not memory_stop(output):
            state.update(status='needs_review', error='Non-memory failure', log=str(log.relative_to(ROOT)))
            save()
            raise RuntimeError('PAM stopped for non-memory failure; inspect ' + str(log))
        state.update(status='waiting_after_memory_stop', log=str(log.relative_to(ROOT)))
        save()
        time.sleep(120)
    state.update(status='needs_review', error='24-hour wait budget or 12-attempt limit reached')
    save()
    raise RuntimeError(state['error'])


if __name__ == '__main__':
    main()
