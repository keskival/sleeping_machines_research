"""Source-exact recovery of a saved sparse-validation ladder; stdlib preparation.

Historical sources run only on the declared physical host, inside run_safe's
inherited reservation. No fitted weights or live source files are changed.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shlex
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 12 * 1024 * 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p) != name:
        raise ValueError('Unsafe relative path: ' + str(name))
    return name


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def git_bytes(revision, name):
    return subprocess.run(['git', 'show', revision + ':' + safe_name(name)],
        cwd=ROOT, check=True, capture_output=True, timeout=10).stdout


def source_closure(revision, seeds):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Exact historical commit required')
    names = subprocess.run(['git', 'ls-tree', '-r', '--name-only', revision],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()
    python = {n for n in names if n.endswith('.py')}
    pending, files = set(seeds), {}

    def resolve(module):
        base = module.replace('.', '/')
        return {candidate for prefix in ('', 'experiments/', 'scripts/')
                for candidate in (prefix + base + '.py', prefix + base + '/__init__.py')
                if candidate in python}

    while pending:
        name = pending.pop()
        if name in files:
            continue
        payload = git_bytes(revision, name)
        files[name] = payload
        if sum(map(len, files.values())) > MAX_BYTES:
            raise ValueError('Historical source closure exceeds 12 MiB')
        tree = ast.parse(payload, filename=name)
        package = list(PurePosixPath(name).parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                parts = prefix + (node.module.split('.') if node.module else [])
                base = '.'.join(parts)
                modules = [base] + ['.'.join(parts + [a.name]) for a in node.names if a.name != '*']
            else:
                continue
            for module in modules:
                pending.update(resolve(module) - files.keys())
        for parent in PurePosixPath(name).parents:
            init = str(parent / '__init__.py')
            if init in python and init not in files:
                pending.add(init)
    return files


def prepare(original, revision, tag):
    if not re.fullmatch(r'aws_[A-Za-z0-9_]+', tag):
        raise ValueError('Fresh AWS-specific tag required')
    ladder_path = ROOT / safe_name(original)
    ladder = json.loads(ladder_path.read_text())
    if ladder['status'] != 'prepared_unrun' or len(ladder['jobs']) != 7:
        raise ValueError('Original seven-stage unused ladder required')
    parent_path = ROOT / safe_name(ladder['parent'])
    parent = json.loads(parent_path.read_text())
    checkpoint = ROOT / safe_name(parent['final_weights'])
    if parent['status'] != 'completed' or sha(parent_path) != ladder['parent_sha256']:
        raise ValueError('Changed completed fit')
    if sha(checkpoint) != ladder['weights_sha256']:
        raise ValueError('Actual saved checkpoint required')
    desired = dict(parent['source_sha256'])
    controls = {original: ladder_path.read_bytes(), ladder['parent']: parent_path.read_bytes()}
    for job in ladder['jobs']:
        spec_path = ROOT / safe_name(job['manifest'])
        spec = json.loads(spec_path.read_text())
        queue = ROOT / safe_name(job['queue'])
        if sha(queue) != spec['queue_sha256'] or (ROOT / safe_name(job['output'])).exists():
            raise ValueError('Changed queue or an already executed original stage')
        for name, digest in spec['source_sha256'].items():
            if name in desired and desired[name] != digest:
                raise ValueError('Incompatible historical source pins')
            desired[name] = digest
        controls[job['manifest']] = spec_path.read_bytes()
        controls[job['queue']] = queue.read_bytes()
    files = source_closure(revision, set(desired) | {'experiments/check_sparse_rescore_protocol.py'})
    for name, expected in desired.items():
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError('Historical revision fails a frozen source pin: ' + name)
    directory = ROOT / 'experiments/queue' / tag
    directory.mkdir(exist_ok=False)
    archive = directory / 'historical_sources.zip'
    content = dict(files, **controls)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for name, payload in sorted(content.items()):
            stream.writestr(safe_name(name), payload)
    jobs = []
    for index, old in enumerate(ladder['jobs']):
        stage = 'contracts' if index == 0 else f"{old['stage']}_{old['split']}_T{old['segment']}"
        name = tag + '_' + stage
        queue = directory / (name + '.txt')
        output = f'experiments/results/diagnostics/{name}.json'
        queue.write_text('# PREPARED UNRUN; AWS physical reservation, original owners first.\n'
            '# Use run_safe only, with the declared guards. No training or optimizer.\n'
            f"# MEM_CAP_KB={old['guards']['address_space_kib']} MEM_CAP_RSS_KB={old['guards']['rss_kib']} "
            f"MIN_AVAIL_MB={old['guards']['min_available_mib']} JOB_TIMEOUT_S={old['guards']['timeout_s']}\n"
            f'{name} experiments/sparse_validation_recovery.py run --manifest '
            f'experiments/queue/{tag}/manifest.json --stage {stage}\n')
        jobs.append(dict(stage=stage, original=old, queue=str(queue.relative_to(ROOT)),
                         queue_sha256=sha(queue), output=output, guards=old['guards'],
                         required_stages=[] if index == 0 else ['contracts'] +
                         ([f"prefix_dev_T{old['segment']}"] if old['stage'] == 'full' else [])))
    record = dict(status='prepared_unrun', host='ip-172-31-47-132', tag=tag,
        historical_revision=revision, original_manifest=original,
        original_manifest_sha256=sha(ladder_path),
        parent=ladder['parent'], parent_sha256=ladder['parent_sha256'],
        checkpoint=parent['final_weights'], checkpoint_sha256=ladder['weights_sha256'],
        dataset='data/text8/text8', dataset_sha256=sha(ROOT/'data/text8/text8'),
        archive=str(archive.relative_to(ROOT)), archive_sha256=sha(archive),
        archived_sha256={n: hashlib.sha256(b).hexdigest() for n,b in sorted(content.items())},
        frozen_source_pins=desired, source_files=len(files), jobs=jobs,
        launcher_sha256=sha(Path(__file__)),
        current_source_drift=[n for n,h in desired.items() if sha(ROOT/n) != h],
        scope='Exact saved producer/validator sources, restored in an isolated root. '
              'No original manifest, checkpoint or active source overwritten. '
              'Native contracts and quality remain unrun. Dataset binding is new, '
              'not retroactive proof of the historical dataset bytes.')
    dump(directory/'manifest.json', record)
    return record


def verify_record(record):
    if sha(Path(__file__)) != record['launcher_sha256']:
        raise ValueError('Changed recovery launcher')
    for name, expected in ((record['archive'], record['archive_sha256']),
            (record['original_manifest'], record['original_manifest_sha256']),
            (record['parent'], record['parent_sha256']),
            (record['checkpoint'], record['checkpoint_sha256']),
            (record['dataset'], record['dataset_sha256'])):
        if sha(ROOT/safe_name(name)) != expected:
            raise ValueError('Changed recovery input: ' + name)
    for job in record['jobs']:
        if sha(ROOT/safe_name(job['queue'])) != job['queue_sha256']:
            raise ValueError('Changed guarded recovery queue')


def materialize(record, target):
    """Recover sources/control bytes without importing any model or unpickling."""
    with zipfile.ZipFile(ROOT/record['archive']) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or set(names) != set(record['archived_sha256']):
            raise ValueError('Unexpected or duplicated archive members')
        if sum(i.file_size for i in stream.infolist()) > MAX_BYTES:
            raise ValueError('Archive exceeds the 12 MiB boundary')
        for name in names:
            safe_name(name)
            payload = stream.read(name)
            if hashlib.sha256(payload).hexdigest() != record['archived_sha256'][name]:
                raise ValueError('Changed historical archive member')
            output = target/name
            if output.exists():
                if output.is_symlink() or sha(output) != record['archived_sha256'][name]:
                    raise ValueError('Changed materialized historical file')
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('xb') as handle:
                    handle.write(payload)
    weights = target/record['checkpoint']
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        with weights.open('xb') as destination, (ROOT/record['checkpoint']).open('rb') as source:
            shutil.copyfileobj(source, destination, 1024*1024)
    if weights.is_symlink() or sha(weights) != record['checkpoint_sha256']:
        raise ValueError('Changed isolated actual checkpoint')
    (target/'experiments/results/diagnostics').mkdir(parents=True, exist_ok=True)
    return target


def physical_reservation(record):
    if Path('/.dockerenv').exists() or os.uname().nodename != record['host']:
        raise ValueError('The declared physical AWS host is required')
    # run_safe opens FD9 before launching its child. Reusing its locked open
    # description succeeds; finding someone else's lock is insufficient.
    slot = os.environ.get('AWS_GYM_SLOT')
    expected = '/tmp/experiments-runner.lock'
    if slot:
        if slot not in ('1', '2', '3'):
            raise ValueError('Invalid bounded AWS slot')
        expected = f'/tmp/experiments-runner.aws-gym-slot{slot}.lock'
        fd = int(os.environ['AWS_GYM_HOST_LOCK_FD'])
        if os.readlink(f'/proc/self/fd/{fd}') != '/tmp/experiments-runner.lock':
            raise ValueError('Inherited ordinary host reservation required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.readlink('/proc/self/fd/9') != expected:
        raise ValueError('Inherited run_safe reservation required')
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)


def verify_guards(job, environment, address_space, available_kib):
    guards = job['guards']
    for key, expected in (('MEM_CAP_RSS_KB', guards['rss_kib']),
            ('MIN_AVAIL_MB', guards['min_available_mib']), ('JOB_TIMEOUT_S', guards['timeout_s'])):
        if environment.get(key) != str(expected):
            raise ValueError('Use the declared run_safe guard: ' + key)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCH_NUM_THREADS'):
        if environment.get(key) != '1':
            raise ValueError('One-thread guarded execution required')
    if address_space != guards['address_space_kib']*1024:
        raise ValueError('Declared run_safe address-space cap required')
    if available_kib < guards['min_available_mib']*1024:
        raise ValueError('Available-memory admission floor failed')


def admit_predecessors(record, job, manifest_hash, runtime, root=ROOT):
    for stage in job['required_stages']:
        previous, = [j for j in record['jobs'] if j['stage'] == stage]
        output = root/safe_name(previous['output'])
        sidecar = json.loads(output.with_suffix('.recovery.json').read_text())
        if (sidecar.get('status') != 'completed' or sidecar.get('exit_code') != 0
                or sidecar.get('manifest_sha256') != manifest_hash
                or sidecar.get('stage') != stage or sidecar.get('output_sha256') != sha(output)):
            raise ValueError('A published recovery predecessor has not passed: ' + stage)
        if sha(runtime/safe_name(previous['original']['output'])) != sha(output):
            raise ValueError('Isolated predecessor differs from published evidence')


def run(manifest_name, stage):
    manifest_path = ROOT/safe_name(manifest_name)
    record = json.loads(manifest_path.read_text())
    physical_reservation(record)  # before file extraction or native imports
    verify_record(record)
    manifest_hash = sha(manifest_path)
    job, = [j for j in record['jobs'] if j['stage'] == stage]
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:')))
    verify_guards(job, os.environ, resource.getrlimit(resource.RLIMIT_AS)[0], available)
    output = ROOT/safe_name(job['output'])
    sidecar = output.with_suffix('.recovery.json')
    if output.exists() or sidecar.exists():
        raise ValueError('Preserve previously completed recovery outputs')
    runtime = materialize(record, ROOT/'.git/sparse-validation-recovery'/record['tag'])
    admit_predecessors(record, job, manifest_hash, runtime)
    original = job['original']
    queue = runtime/original['queue']
    line, = [s for s in queue.read_text().splitlines() if s.strip() and not s.startswith('#')]
    tokens = shlex.split(line)
    source_output = runtime/original['output']
    if source_output.exists():
        raise ValueError('Original stage already ran in this isolated root; preserve it')
    environment = dict(os.environ, PYTHONPATH=str(runtime)+os.pathsep+str(runtime/'experiments'),
                       PYTHONDONTWRITEBYTECODE='1')
    started = datetime.now(timezone.utc).isoformat()
    child = subprocess.run([sys.executable, str(runtime/safe_name(tokens[1])), *tokens[2:]],
                           cwd=ROOT, env=environment)
    verify_record(record)
    for name, digest in record['archived_sha256'].items():
        if sha(runtime/name) != digest:
            raise ValueError('Materialized historical source/control changed during execution')
    if sha(manifest_path) != manifest_hash:
        raise ValueError('Recovery manifest changed during execution')
    native = json.loads(source_output.read_text()) if source_output.exists() else None
    exit_code = child.returncode
    if exit_code == 0 and (not native or native.get('status') != 'completed'
                          or (stage == 'contracts' and native.get('contracts_admitted') is not True)):
        exit_code = 2
    if source_output.exists():
        with output.open('xb') as destination, source_output.open('rb') as source:
            shutil.copyfileobj(source, destination)
    dump(sidecar, dict(status='completed' if exit_code == 0 else 'failed',
        exit_code=exit_code, child_exit_code=child.returncode, started_utc=started,
        manifest_sha256=manifest_hash, stage=stage,
        historical_revision=record['historical_revision'],
        output_sha256=sha(output) if output.exists() else None,
        scope='Historical numerical result copied unchanged; run_safe log includes all work.'))
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--original', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--tag', required=True)
    r = commands.add_parser('run')
    r.add_argument('--manifest', required=True)
    r.add_argument('--stage', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.original, args.revision, args.tag)
        print(json.dumps(dict(status=result['status'], source_files=result['source_files'],
                             drift=result['current_source_drift'], jobs=len(result['jobs']))))
    else:
        raise SystemExit(run(args.manifest, args.stage))
