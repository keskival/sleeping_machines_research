"""B3 cross-host identity: exact split bytes, with generation timers excluded.

Both raw manifest digests are bound. Only top-level started and per-split
wall_s are excluded from semantic equality; every other field must agree.
"""
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_identity(local_path, reference_path, expected_local, expected_reference):
    if digest(local_path)!=expected_local or digest(reference_path)!=expected_reference:
        raise ValueError('Raw fitted/local data manifest digest differs')
    def payload(path):
        value=copy.deepcopy(json.loads(Path(path).read_text()))
        value.pop('started',None)
        for split in value['splits'].values():split.pop('wall_s',None)
        return value
    local,reference=payload(local_path),payload(reference_path)
    if local!=reference:
        raise ValueError('Frozen data payload/recipe differs beyond generation timers')
    required={'train_clean','val_clean','val_faulty','test_clean','test_faulty'}
    if set(local['splits'])!=required or any('sha256' not in row for row in local['splits'].values()):
        raise ValueError('Five complete frozen split identities required')
    return hashlib.sha256(json.dumps(local,sort_keys=True,separators=(',',':')).encode()).hexdigest()
