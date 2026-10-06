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


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slot', required=True, choices=['1', '2', '3'])
    ap.add_argument('--job', action='append', required=True, help='"tag|race_tpp args" ; tag is also the job name')
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--rss-kb', type=int, default=3000000)
    ap.add_argument('--purpose', required=True, help='the decision this batch of results changes')
    ap.add_argument('--source', default='experiments/tpp/race_tpp.py', help='pinned driver version')
    ap.add_argument('--dep', action='append', default=[], help='additional pinned source files')
    ap.add_argument('--battle', default='B1')
    ap.add_argument('--result-dir', default='experiments/results/tpp')
    a = ap.parse_args()
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    jobs = []
    for spec in a.job:
        tag, args = spec.split('|', 1)
        qdir = ROOT / 'experiments/queue' / ('b1_easytpp' if a.battle == 'B1' else a.battle.lower()) / tag
        qdir.mkdir(parents=True, exist_ok=True)
        q = qdir / f'{tag}.txt'
        if q.exists():
            raise SystemExit(f'queue {q} exists; job names are never reused')
        q.write_text(f'# {a.battle}: {a.purpose}\n{tag} {a.source} --tag {tag} {args}\n')
        rel = str(q.relative_to(ROOT))
        jobs.append(dict(name=tag, queue=rel, queue_sha256=sha(rel), kind=f'{a.battle.lower()}_job',
                         source_sha256={s: sha(s) for s in [a.source] + a.dep},
                         result=f'{a.result_dir}/{tag}.json', timeout_s=a.timeout,
                         rss_kb=a.rss_kb, vms_kb=8000000))
    packet = dict(host='ip-172-31-47-132', battle=a.battle, purpose=a.purpose,
                  parent_manifest_sha256=sha(str((COORD / 'manifest.json').relative_to(ROOT))),
                  slots={a.slot: jobs})
    out = COORD / 'addenda' / f'zzzzzzzzzzzzzzzzzzzzzzzzzzz{a.battle.lower()}_{stamp}_s{a.slot}.json'
    out.write_text(json.dumps(packet, indent=2) + '\n')
    print(out.relative_to(ROOT))
    for j in jobs:
        print(' ', j['name'], '->', j['result'])


if __name__ == '__main__':
    main()
