"""Resolve an explicitly declared same-settings retry without replacing history."""
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT/'experiments/queue/tuned_reference_recoveries.json'


def result_directory(queue, arguments, result_root):
    queue, result_root = Path(queue), Path(result_root)
    original = result_root/queue.stem
    if not BINDINGS.exists():
        return original
    for row in json.loads(BINDINGS.read_text())['recoveries']:
        if queue.resolve() != (ROOT/row['original_queue']).resolve():
            continue
        def tokens(path, digest):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Changed recovery queue: '+str(path))
            lines = [s for s in path.read_text().splitlines() if s and not s.startswith('#')]
            if len(lines) != 1:
                raise ValueError('One-job recovery required')
            return shlex.split(lines[0])
        before = tokens(queue, row['original_queue_sha256'])
        after = tokens(ROOT/row['recovery_queue'], row['recovery_queue_sha256'])
        if before[before.index('--')+1:] != arguments or after[after.index('--')+1:] != arguments:
            raise ValueError('Recovery changes numerical arguments')
        if before[before.index('--script')+1] != after[after.index('--script')+1]:
            raise ValueError('Recovery changes numerical script')
        for command in (before, after):
            if command[0] != command[command.index('--run-tag')+1]:
                raise ValueError('Recovery job/output identity differs')
        target = result_root/after[0]
        if target.parent != result_root:
            raise ValueError('Recovery must remain inside result root')
        prior = original/'provenance.json'
        if prior.exists() and json.loads(prior.read_text()).get('status') == 'completed':
            return original
        provenance = target/'provenance.json'
        if not provenance.exists():
            return original
        saved = json.loads(provenance.read_text())
        if saved.get('status') != 'completed':
            return original
        if saved.get('source_sha256') != row['script_sha256']:
            raise ValueError('Recovery numerical source differs')
        if saved.get('arguments') != arguments:
            raise ValueError('Recovery recorded arguments differ')
        return target
    return original
