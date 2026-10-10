"""Source-bound B4 numerical development comparison; no test scoring."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    p.add_argument('--fit', required=True)
    a = p.parse_args()
    paths = [a.fit, 'experiments/results/tpp_b4/b4g_github_s1.json']
    records = [json.loads((ROOT/path).read_text()) for path in paths]
    for r in records:
        for source, sha in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == sha, source
    fit, old = records
    assert fit['stage'] == 'logspread_development_fit' and 'test' not in fit
    assert fit['config'] == old['config']
    successful = fit['status'] == 'completed' and fit['skipped_updates'] == 0
    delta = old['val']['total']-fit['best_dev_total'] if successful else None
    curves = [dict(epoch=x['epoch'], repaired_dev=x['val']['total'], guarded_dev=y['val_total'])
              for x,y in zip(fit['history'],old['history']) if x['epoch'] == y['epoch']]
    result = dict(status='completed', battle='B4 development', tag=a.tag,
                  stage='numerical_development_decision', finite_update_gate=successful,
                  repaired_best_dev_total=fit['best_dev_total'], guarded_best_dev_total=old['val']['total'],
                  repaired_dev_loss_reduction=delta, development_quality_improves=delta is not None and delta > 0,
                  repaired_epochs=fit['epochs_run'], guarded_epochs=old['epochs_run'],
                  repaired_skips=fit['skipped_updates'], guarded_skips=old['skipped_updates'],
                  fitting_wall_s=fit['fitting_wall_s'], evaluation_wall_s=fit['evaluation_wall_s'],
                  completed_train_targets=fit['completed_train_targets'], inference=fit['inference'],
                  common_epoch_dev_curve=curves,
                  decision='Use completed DEV comparison and skipped-update diagnosis to choose the next numerical/mark-path step; no automatic fit.',
                  scope='One Github split1 seed0; own-model DEV comparison. Original guarded fitting skipped padded batches. '
                        'Each arm selected by its DEV patience within its saved budget; frozen TEST verdict unchanged. No published-reference or matched-compute win.',
                  inputs={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
                  source_sha256={'experiments/tpp/summarize_b4_development.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out = ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as stream: stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
