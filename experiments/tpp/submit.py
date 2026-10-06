#!/usr/bin/env python3
"""Submit B1 jobs to the AWS coordinator as an immutable addendum (one queue file per job, run through run_safe.sh).

    python experiments/tpp/submit.py --slot 1 --name b1_taxi_dev1 -- --dataset taxi --tag ... [race_tpp args]

Repeat --job to put several jobs on one slot in order. Every job is battle B1; the result file is the run's JSON.
"""
import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COORD = ROOT / 'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
SOURCES = ['experiments/tpp/race_tpp.py']


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slot', required=True, choices=['1', '2', '3'])
    ap.add_argument('--job', action='append', required=True, help='"tag|race_tpp args" ; tag is also the job name')
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--rss-kb', type=int, default=3000000)
    ap.add_argument('--purpose', required=True, help='the decision this batch of results changes')
    a = ap.parse_args()
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    jobs = []
    for spec in a.job:
        tag, args = spec.split('|', 1)
        qdir = ROOT / 'experiments/queue/b1_easytpp' / tag
        qdir.mkdir(parents=True, exist_ok=True)
        q = qdir / f'{tag}.txt'
        if q.exists():
            raise SystemExit(f'queue {q} exists; job names are never reused')
        q.write_text(f'# B1 EasyTPP: {a.purpose}\n{tag} experiments/tpp/race_tpp.py --tag {tag} {args}\n')
        rel = str(q.relative_to(ROOT))
        jobs.append(dict(name=tag, queue=rel, queue_sha256=sha(rel), kind='b1_tpp',
                         source_sha256={s: sha(s) for s in SOURCES},
                         result=f'experiments/results/tpp/{tag}.json', timeout_s=a.timeout,
                         rss_kb=a.rss_kb, vms_kb=8000000))
    packet = dict(host='ip-172-31-47-132', battle='B1', purpose=a.purpose,
                  parent_manifest_sha256=sha(str((COORD / 'manifest.json').relative_to(ROOT))),
                  slots={a.slot: jobs})
    out = COORD / 'addenda' / f'zzzzzzzzzzzzzzzzzzzzzzzzzzzb1_{stamp}_s{a.slot}.json'
    out.write_text(json.dumps(packet, indent=2) + '\n')
    print(out.relative_to(ROOT))
    for j in jobs:
        print(' ', j['name'], '->', j['result'])


if __name__ == '__main__':
    main()
