"""Guarded native retry below a frozen tuned Transformer fitting-work budget."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from language_kwrite_protocol_v2 import digest, preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--stage', required=True)
    args = p.parse_args()
    manifest, cfg = preflight(args.manifest, args.stage)
    reference = manifest['reference']
    if digest(ROOT / reference['path']) != reference['sha256']:
        raise ValueError('Frozen tuned reference changed')
    import language_batched_benchmark as base
    sys.argv = [str(ROOT / 'experiments/language_batched_benchmark.py'), *cfg['arguments']]
    base.main()
    out = ROOT / cfg['output']
    result = json.loads(out.read_text())
    actual = result['work']['whole_fit_unit_special_flops_estimate']
    if not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual <= 0:
        raise ValueError('Finite positive actual fitting work required')
    geometry = (result.get('eval_segment') == 256 and
                result.get('test_targets_eval_segment') == 999936 and
                result.get('protocol', {}).get('test') == [95000000, 96000000])
    quality = result.get('test_bpc_eval_segment')
    finite_quality = isinstance(quality, (int, float)) and math.isfinite(quality)
    result.update(stage=args.stage, manifest_sha256=digest(args.manifest),
                  source_sha256=manifest['source_sha256'], reference=reference,
                  budget_status='within_budget' if actual <= reference['whole_fit_flops'] else 'over_budget',
                  benchmark_eligible=cfg['kind'] == 'fit' and actual <= reference['whole_fit_flops']
                                     and geometry and finite_quality,
                  protocol_geometry_matches=geometry)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(stage=args.stage, budget_status=result['budget_status'],
                         benchmark_eligible=result['benchmark_eligible'])))


if __name__ == '__main__':
    main()
