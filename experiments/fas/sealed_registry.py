"""B3 once-only scoring reservation and atomic evidence publication.

Reservations survive interruptions. An interrupted scoring requires explicit
reconciliation; neither changing an output tag nor rerunning clears it.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def reserve(directory, tag, identity, provenance):
    """Reserve both the original trained-model tag and checkpoint/config identity."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if Path(tag).name != tag or not tag or tag in ('.', '..'):
        raise ValueError('A plain trained-model tag is required')
    with (directory/'fas_v2_test_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = directory/'fas_v2_test_ledger.jsonl'
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                record = json.loads(line)
                if record.get('tag') == tag or record.get('identity') == identity:
                    raise ValueError('Model has an existing TEST ledger entry')
        if any(json.loads(p.read_text()).get('identity') == identity
               for p in directory.glob('*_TEST_started.json')):
            raise ValueError('Model already has a TEST reservation, including interrupted scoring')
        reservation = directory/f'{tag}_TEST_started.json'
        if reservation.exists() or (directory/f'{tag}_TEST.json').exists() or (directory/f'{tag}_TEST_scores.npz').exists():
            raise FileExistsError('Existing TEST reservation or output requires reconciliation')
        record = dict(provenance, tag=tag, identity=identity, status='reserved',
                      utc=datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'))
        # Exclusive creation is durable before any test arrays are loaded.
        with reservation.open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        sync_directory(directory)
        return reservation


def complete(reservation, result_path, score_path):
    reservation = Path(reservation); record = json.loads(reservation.read_text())
    entry = dict(record, status='completed', result_sha256=sha(result_path), scores_sha256=sha(score_path),
                 completed_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'))
    directory = reservation.parent
    with (directory/'fas_v2_test_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = directory/'fas_v2_test_ledger.jsonl'
        if ledger.exists() and any(json.loads(line).get('identity') == entry['identity']
                                   for line in ledger.read_text().splitlines()):
            raise ValueError('Completion already recorded')
        with ledger.open('a') as stream:
            stream.write(json.dumps(entry)+'\n'); stream.flush(); os.fsync(stream.fileno())
    # Keep the original reservation permanently; the completed ledger resolves it.
    return entry
