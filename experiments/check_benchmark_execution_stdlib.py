"""Dispatch, queue, provenance and physical-admission contracts; no ML imports."""
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    wrapper = runpy.run_path(str(ROOT / 'experiments/language_kwrite_protocol_v2.py'))
    status = runpy.run_path(str(ROOT / 'experiments/benchmark_execution_status.py'))
    calls = []

    def recruit(*args, **kw):
        calls.append(kw)
        return tuple(range(8))

    ce = SimpleNamespace(compiled_logits=lambda *a, **kw: kw['step']('dummy'))
    base = SimpleNamespace(batched_logits='old tracing policy')
    compiled = []

    def compiler(fn, **kw):
        compiled.append(kw)
        return fn

    eager, factory = wrapper['install_policy'](base, ce, recruit,
        SimpleNamespace(compile=compiler), 2, False)
    assert base.batched_logits(None, [], 1, all_logits=True, route_credit='linear') == tuple(range(7))
    assert ce.compiled_step()('dummy') == tuple(range(7))
    assert ce.compiled_step()('dummy') == tuple(range(7))
    assert all(c == dict(write_k=2, deliver_k=False) for c in calls)
    assert len(compiled) == 1 and compiled[0] == dict(dynamic=False, fullgraph=True)
    groups = ['same_policy_trace_compile_evaluation_dispatch']
    with tempfile.TemporaryDirectory(prefix='benchmark-check-') as directory:
        root = Path(directory)
        q = root / 'queue.txt'
        q.write_text('job driver.py --tag result\n')
        task = dict(id='sample', owner='fixture', queue='queue.txt',
                    queue_sha256=status['sha'](q), result='result.json')
        assert status['check'](task, root)['status'] == 'not_completed'
        (root / 'result.json').write_text(json.dumps(dict(status='running')))
        assert status['check'](task, root)['status'] == 'reported_running'
        (root / 'result.json').write_text(json.dumps(dict(status='completed', args={'tag':'wrong'})))
        assert status['check'](task, root)['status'] == 'result_arguments_mismatch'
        (root / 'result.json').write_text(json.dumps(dict(status='completed', args={'tag':'result'})))
        assert status['check'](task, root)['status'] == 'completed'
        q.write_text('changed driver.py\n')
        assert status['check'](task, root)['status'] == 'queue_changed'
        groups.append('queue_presence_running_result_arguments_and_drift')
        q.write_text('job driver.py --tag result\n')
        task['stage_manifest'] = 'manifest.json'; task['stage'] = 'fit'
        manifest = dict(status='prepared_unrun', source_sha256={}, stages={
            'smoke':dict(kind='smoke', output='smoke.json'),
            'fit':dict(kind='fit', output='result.json', requires=['smoke'], guards={})})
        mf = root / 'manifest.json'; mf.write_text(json.dumps(manifest))
        (root / 'result.json').unlink()
        assert status['check'](task, root)['status'] == 'prerequisites_pending'
        (root / 'smoke.json').write_text(json.dumps(dict(status='completed', manifest_sha256=status['sha'](mf))))
        assert status['check'](task, root)['status'] == 'not_completed'
        r = dict(status='completed', stage='fit', manifest_sha256=status['sha'](mf),
                 source_sha256={}, benchmark_eligible=False)
        (root / 'result.json').write_text(json.dumps(r))
        assert status['check'](task, root)['status'] == 'completed_not_benchmark_eligible'
        r['benchmark_eligible'] = True
        (root / 'result.json').write_text(json.dumps(r))
        assert status['check'](task, root)['status'] == 'completed'
        r['manifest_sha256'] = 'bad'
        (root / 'result.json').write_text(json.dumps(r))
        assert status['check'](task, root)['status'] == 'result_binding_mismatch'
        groups.append('frozen_prerequisites_result_binding_and_budget_eligibility')
        tasks = []
        for name in ('fit', 'smoke', 'data'):
            qp = root / (name + '.txt')
            qp.write_text(name + ' driver.py\n')
            rp = root / (name + '.json')
            if name != 'data':
                rp.write_text(json.dumps(dict(status='completed')))
            tasks.append(dict(id=name, owner='fixture', queue=qp.name,
                queue_sha256=status['sha'](qp), result=rp.name,
                requires_tasks={'fit':['smoke'], 'smoke':['data'], 'data':[]}[name]))
        report = status['audit'](dict(tasks=tasks), root)
        assert report['completed'] == 0
        assert report['tasks'][0]['status'] == 'completed_prerequisites_unverified'
        (root / 'data.json').write_text(json.dumps(dict(status='completed')))
        assert status['audit'](dict(tasks=tasks), root)['status'] == 'completed'
        tasks[2]['requires_tasks'] = ['fit']
        try:
            status['audit'](dict(tasks=tasks), root)
        except ValueError as error:
            assert 'Cyclic' in str(error)
        else:
            raise AssertionError('Cycle admitted')
        groups.append('recursive_cross_host_prerequisites_and_cycle_rejection')
        # A revised research priority must preserve deferred evidence while
        # distinguishing it from work still required by the active checklist.
        tasks[2]['requires_tasks'] = []
        tasks[2]['disposition'] = 'historical'
        tasks[1]['disposition'] = 'deferred'
        (root / 'smoke.json').unlink()
        revised = status['audit'](dict(tasks=tasks), root)
        assert revised['required'] == 1 and revised['retained'] == 3
        assert revised['deferred'] == 1 and revised['historical_completed'] == 1
        assert revised['status'] == 'incomplete'  # active fit still needs smoke
        assert status['audit'](dict(tasks=[dict(tasks[1], requires_tasks=[])]), root)['status'] == 'no_required_tasks'
        groups.append('explicit_deferral_preserves_evidence_and_active_dependencies')
        external = dict(status='completed', args={'seed':6},
                        work={'whole_fit_unit_special_flops_estimate':99.},
                        test_bpc_eval_segment=1.9, eval_segment=256,
                        test_targets_eval_segment=999936, source_sha256={})
        requirement = dict(output='external.json', expected_args={'seed':6},
                           cap_flops=100., reference_bpc=2.2)
        frozen = dict(status='prepared_unrun', source_sha256={}, host='fixture',
                      stages={'seed7':dict(output='unused.json', requires=[], guards={},
                                          external_requires=[requirement])})
        gate = root / 'gate.json'; gate.write_text(json.dumps(frozen))
        original_root = wrapper['preflight'].__globals__['ROOT']
        wrapper['preflight'].__globals__['ROOT'] = root
        cases = [dict(external), dict(external, test_bpc_eval_segment=2.3),
                 dict(external, test_targets_eval_segment=42),
                 dict(external, work={'whole_fit_unit_special_flops_estimate':101.}),
                 dict(external, source_sha256={'different.py':'bad'})]
        for i, record in enumerate(cases):
            (root / 'external.json').write_text(json.dumps(record))
            try:
                wrapper['preflight'](gate, 'seed7')
            except ValueError as error:
                if i == 0:
                    assert 'physical-host reservation' in str(error)
                else:
                    assert 'win gate' in str(error) or 'producer differs' in str(error)
            else:
                raise AssertionError('External gate or container refusal bypassed')
        wrapper['preflight'].__globals__['ROOT'] = original_root
        groups.append('confirmation_gate_quality_actual_budget_targets_and_sources')
    for folder in ('aws_kwrite_protocol_v2_20261005T104500Z', 'aws_native_strict_tfB_20261005T104500Z'):
        mf = ROOT / 'experiments/queue' / folder / 'manifest.json'
        manifest = json.loads(mf.read_text())
        for p, digest in manifest['source_sha256'].items():
            assert wrapper['digest'](ROOT / p) == digest, p
        for stage in manifest['stages']:
            queue = mf.parent / (folder + '_' + stage + '.txt')
            assert len(status['queue_tokens'](queue)) > 2
        try:
            wrapper['preflight'](mf, next(iter(manifest['stages'])))
        except ValueError as error:
            assert 'physical-host reservation' in str(error), str(error)
        else:
            raise AssertionError('Container admitted before numerical imports')
    groups.append('all_new_source_pins_queues_and_physical_refusal')
    import sys
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='completed', groups=groups, numerical_imports=0,
                          scope='Structural/admission checks only; numerical host contracts remain unrun.')))


if __name__ == '__main__':
    main()
