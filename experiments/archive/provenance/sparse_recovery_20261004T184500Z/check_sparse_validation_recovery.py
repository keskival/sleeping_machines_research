"""Historical-source and admission checks only; never import a tensor runtime."""
import argparse
import builtins
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from sparse_validation_recovery import (ROOT, materialize, physical_reservation,
                                       safe_name, sha, verify_guards, verify_record)


def reject(function):
    try:
        function()
    except (ValueError, OSError, KeyError):
        return
    raise AssertionError('Invalid recovery input accepted')


def check(manifest_name):
    manifest_path = ROOT/safe_name(manifest_name)
    record = json.loads(manifest_path.read_text())
    verify_record(record)
    for bad in ('../escape', '/absolute', 'a/../b', './a', ''):
        reject(lambda: safe_name(bad))
    guards = record['jobs'][0]['guards']
    environment = dict(MEM_CAP_RSS_KB=str(guards['rss_kib']),
        MIN_AVAIL_MB=str(guards['min_available_mib']), JOB_TIMEOUT_S=str(guards['timeout_s']),
        **{n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCH_NUM_THREADS')})
    address_space = guards['address_space_kib']*1024
    available = guards['min_available_mib']*1024
    verify_guards(record['jobs'][0], environment, address_space, available)
    invalid = 0
    for name in environment:
        bad = dict(environment)
        bad[name] = '2' if name.endswith('THREADS') else '0'
        reject(lambda: verify_guards(record['jobs'][0], bad, address_space, available))
        invalid += 1
    reject(lambda: verify_guards(record['jobs'][0], environment, -1, available))
    reject(lambda: verify_guards(record['jobs'][0], environment, address_space, available-1))
    invalid += 2
    # This workspace is not the physical host; never pretend a local lock admits it.
    if Path('/.dockerenv').exists() or os.uname().nodename != record['host']:
        reject(lambda: physical_reservation(record))
    else:
        raise ValueError('Run this stdlib check in the non-admitted review workspace')
    with tempfile.TemporaryDirectory(prefix='sparse-recovery-check-', dir=ROOT/'.git') as folder:
        target = materialize(record, Path(folder))
        materialize(record, target)  # source-exact restart, without overwriting
        for name, digest in record['frozen_source_pins'].items():
            assert sha(target/name) == digest
        before = {name: sha(ROOT/name) for name in record['frozen_source_pins']}
        corrupted = target/'sleeping_machines/batched_episodes.py'
        payload = corrupted.read_bytes()
        corrupted.write_bytes(payload + b'\n# changed\n')
        reject(lambda: materialize(record, target))
        corrupted.write_bytes(payload)
        # Exercise the actual recovered contract's complete stdlib preflight,
        # then deliberately stop at its FIRST native import. No pickle/array/model.
        class RuntimeBoundary(Exception):
            pass
        imports = []
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'torch'):
                imports.append(name)
                raise RuntimeBoundary(name)
            return original_import(name, *args, **kwargs)
        sys.path.insert(0, str(target/'experiments'))
        try:
            runner = runpy.run_path(str(target/'experiments/prepacked_sparse_contracts.py'))['run']
            contract = json.loads((target/record['jobs'][0]['original']['manifest']).read_text())
            builtins.__import__ = guarded_import
            try:
                runner(SimpleNamespace(**contract['arguments']))
            except RuntimeBoundary:
                pass
            else:
                raise AssertionError('The actual contract did not reach the native import boundary')
        finally:
            builtins.__import__ = original_import
            sys.path.pop(0)
        assert imports == ['numpy']
        assert before == {name: sha(ROOT/name) for name in before}
        prefix = record['jobs'][1]['original']['manifest']
        result = subprocess.run([sys.executable, str(target/'experiments/check_sparse_rescore_protocol.py'), prefix],
            cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(target)+os.pathsep+str(target/'experiments'),
                              PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True, text=True, check=True, timeout=30)
        inherited = json.loads(result.stdout)
        assert inherited['native_runtime'] == 'unrun'
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return dict(status='passed', scope='Stdlib provenance/import/guard contracts; no numerical evidence',
        recovery_manifest=manifest_name, recovery_manifest_sha256=sha(manifest_path),
        historical_revision=record['historical_revision'],
        archived_dependency_sources=record['source_files'], restored_frozen_source_pins=len(record['frozen_source_pins']),
        guarded_one_job_queues=len(record['jobs']), old_source_drift=record['current_source_drift'],
        actual_trained_preflight_reaches_blocked_native_import=True,
        invalid_guard_cases=invalid, physical_host_refusal=True, corrupt_materialization_refused=True,
        original_queues_and_live_sources_unchanged=True,
        inherited_window_enumerations=inherited['window_enumerations'],
        inherited_invalid_contract_rejections=inherited['invalid_contracts_rejected'],
        inherited_all_seven_queue_source_checks=True,
        native_forward_calls=0, backward_calls=0, optimizer_updates=0,
        source_sha256={n: sha(ROOT/n) for n in ('experiments/sparse_validation_recovery.py',
                                              'experiments/check_sparse_validation_recovery.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS, (1_000_000*1024, 1_000_000*1024))
    result = check(args.manifest)
    with (ROOT/safe_name(args.out)).open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))
