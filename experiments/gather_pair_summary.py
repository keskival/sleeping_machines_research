"""Publish a completed matched-recipe gather pair; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    rows = []
    parents = []
    sources = {}
    for kind in ('control', 'gather'):
        path = Path(f'experiments/results/diagnostics/curie_{kind}_tokens_8k_p24_work_20261006_v1.json')
        audit = json.loads(path.read_text())
        parent_path = Path(audit['control'])
        parent = json.loads(parent_path.read_text())
        assert audit['status'] == parent['status'] == 'completed'
        assert audit['curve_parity_max_error'] < 2e-6
        for phase in ('fitting', 'inference'):
            assert audit[phase]['formula_coverage_complete']
            assert not audit[phase]['unsupported_floating_operators']
        parents.append(parent)
        for source in (path, parent_path):
            sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows.append(dict(execution=kind, selected_dev_nll=audit['selected_dev_nll'],
                         fit_targets=audit['fitting_targets'], inference_targets=audit['inference_targets'],
                         whole_fit_gflops=audit['fitting']['arithmetic_flops']/1e9,
                         fit_mflops_per_target=audit['fitting_arithmetic_flops_per_target']/1e6,
                         inference_mflops_per_target=audit['inference_arithmetic_flops_per_target']/1e6,
                         fit_wall_s=parent['train_wall_s_total'], total_wall_s=parent['wall_s']))
    a, b = parents
    assert {k:v for k,v in a['args'].items() if k != 'tag'} == {k:v for k,v in b['args'].items() if k != 'tag'}
    for key in ('train_sha256', 'dev_sha256'):
        assert a['identity'][key] == b['identity'][key]
    assert a['presentations_total'] == b['presentations_total']
    error = max(abs(x['dev_nll']-y['dev_nll']) for x,y in zip(a['curve'], b['curve'], strict=True))
    record = dict(status='completed', rows=rows, source_sha256=sources,
                  between_execution_curve_max_error=error,
                  fitting_arithmetic_reduction_fraction=1-rows[1]['whole_fit_gflops']/rows[0]['whole_fit_gflops'],
                  fitting_wall_reduction_fraction=1-rows[1]['fit_wall_s']/rows[0]['fit_wall_s'],
                  scope='One paired seed6 8K proper-token pilot. Each full audit replays its own parent. Complete fitting arithmetic boundary excludes initialization/evaluation/serialization; special functions separate and random work unquantified. Inference includes scorer reductions. Wall is one sequential pair, not repeated benchmark. Different later curves; no quality supremacy, physical traffic or energy claim.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
