"""Read-only execution checklist: source/queue/dependency checks, never training.

A queue is not a running job; completion requires a matching result. Physical
locks/occupancy are deliberately not inferred from this review workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def queue_tokens(path):
    lines = [l.strip() for l in Path(path).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith('#')]
    if len(lines) != 1:
        raise ValueError('Each task must use a one-job queue')
    return shlex.split(lines[0])


def check(task, root=ROOT):
    root = Path(root)
    row = dict(id=task['id'], owner=task['owner'], queue=task['queue'],
               result=task['result'], status='not_completed', blockers=[])
    row['disposition'] = task.get('disposition', 'required')
    if row['disposition'] not in ('required', 'deferred', 'historical'):
        raise ValueError('Unknown task disposition')
    if task.get('disposition_reason'):
        row['disposition_reason'] = task['disposition_reason']
    if task.get('kind') == 'diagnostic_spec':
        row['status'] = 'owner_preparation_required'
        row['specification'] = task['specification']
        record = root / task['result']
        if record.is_file():
            r = json.loads(record.read_text())
            if (r.get('status') == 'completed' and r.get('task_id') == task['id'] and
                    r.get('specification_sha256') == sha(root / task['specification']) and
                    r.get('actual_checkpoint_sha256') and r.get('recorded_policy_parity_passed') is True):
                row['status'] = 'completed'
                row['result_sha256'] = sha(record)
        return row
    q = root / task['queue']
    if not q.is_file() or sha(q) != task['queue_sha256']:
        row['status'] = 'queue_changed'
        return row
    tokens = queue_tokens(q)
    frozen = None
    if task.get('stage_manifest'):
        path = root / task['stage_manifest']
        frozen = json.loads(path.read_text())
        cfg = frozen['stages'][task['stage']]
        drift = [p for p, digest in frozen['source_sha256'].items()
                 if not (root / p).is_file() or sha(root / p) != digest]
        row['source_drift'] = drift
        for prior in cfg['requires']:
            parent = root / frozen['stages'][prior]['output']
            if not parent.is_file():
                row['blockers'].append(prior)
                continue
            r = json.loads(parent.read_text())
            if r.get('status') != 'completed' or r.get('manifest_sha256') != sha(path):
                row['blockers'].append(prior)
        row['guards'] = cfg['guards']
    result = root / task['result']
    if not result.is_file():
        if row.get('source_drift'):
            row['status'] = 'source_refresh_required'
        elif row['blockers']:
            row['status'] = 'prerequisites_pending'
        return row
    r = json.loads(result.read_text())
    if r.get('status') != task.get('accepted_status', 'completed'):
        row['status'] = 'reported_' + str(r.get('status', 'unknown'))
        return row
    if frozen:
        if (r.get('stage') != task['stage'] or
                r.get('manifest_sha256') != sha(root / task['stage_manifest']) or
                r.get('source_sha256') != frozen['source_sha256']):
            row['status'] = 'result_binding_mismatch'
            return row
        if frozen['stages'][task['stage']]['kind'] == 'fit' and not r.get('benchmark_eligible'):
            row['status'] = 'completed_not_benchmark_eligible'
            return row
    elif '--tag' in tokens:
        expected = tokens[tokens.index('--tag') + 1]
        if r.get('args', {}).get('tag') != expected:
            row['status'] = 'result_arguments_mismatch'
            return row
    row['status'] = 'completed'
    row['result_sha256'] = sha(result)
    row['metrics'] = {k: r[k] for k in ('test_bpc_eval_segment', 'test_bpc', 'test_auroc',
                                      'memory_half_lives', 'budget_status') if k in r}
    return row


def audit(manifest, root=ROOT):
    rows = [check(t, root) for t in manifest['tasks']]
    by_id = {r['id']: r for r in rows}
    tasks = {t['id']: t for t in manifest['tasks']}
    if len(tasks) != len(rows):
        raise ValueError('Unique task IDs required')
    resolved = set()

    def resolve(identity, visiting):
        if identity in visiting:
            raise ValueError('Cyclic task prerequisites')
        if identity in resolved:
            return
        task, row = tasks[identity], by_id[identity]
        for prior in task.get('requires_tasks', []):
            resolve(prior, visiting | {identity})
            if by_id[prior]['status'] != 'completed':
                row['blockers'].append(prior)
        if row['blockers']:
            if row['status'] == 'completed':
                row['status'] = 'completed_prerequisites_unverified'
            elif row['status'] == 'not_completed':
                row['status'] = 'prerequisites_pending'
        resolved.add(identity)

    for identity in tasks:
        resolve(identity, set())
    active = [r for r in rows if r['disposition'] == 'required']
    return dict(status=('completed' if active and all(r['status'] == 'completed' for r in active)
                        else 'incomplete' if active else 'no_required_tasks'),
                completed=sum(r['status'] == 'completed' for r in active), required=len(active),
                retained=len(rows), deferred=sum(r['disposition'] == 'deferred' for r in rows),
                historical_completed=sum(r['status'] == 'completed' and r['disposition'] == 'historical'
                                         for r in rows), tasks=rows,
                physical_host_inspected=False, numerical_runtime_calls=0,
                scope=('Active execution checklist only; deferred/historical rows remain visible. '
                       'Completion does not establish the research objective or a benchmark win.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    report = audit(manifest)
    report['manifest_sha256'] = sha(args.manifest)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise ValueError('Preserve prior checklist snapshot; use a unique output')
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], completed=report['completed'], required=report['required'],
                         retained=report['retained'], deferred=report['deferred'],
                         historical_completed=report['historical_completed'],
                         tasks=[dict(id=r['id'], status=r['status'], disposition=r['disposition'])
                                for r in report['tasks']]), indent=2))


if __name__ == '__main__':
    main()
