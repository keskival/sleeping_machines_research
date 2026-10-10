"""B4 completed-file comparison of fixed mark-memory and repaired baseline fits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--fit',required=True);a=p.parse_args()
    paths=[a.fit,'experiments/results/tpp_b4/aws_b4_logspread_develop_20261010T2112Z.json']
    records=[json.loads((ROOT/path).read_text()) for path in paths]
    for r in records:
        assert r['status']=='completed' and r['skipped_updates']==0 and 'test' not in r
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
    changed,base=records
    assert changed['stage']=='logspread_marks_development_fit' and base['stage']=='logspread_development_fit'
    assert {k:v for k,v in changed['config'].items() if k not in ('mark_mem','mark_stats')}==base['config']
    assert changed['data_sha256']==base['data_sha256']
    selected=[next(x for x in r['history'] if x['epoch']==r['best_epoch'])['val'] for r in records]
    common=min(len(r['history']) for r in records);window=list(range(max(0,common-20),common))
    assert common>=20
    mark_gain=selected[1]['L_M']-selected[0]['L_M'];total_gain=selected[1]['total']-selected[0]['total']
    window_mark=float(np.mean([base['history'][i]['val']['L_M']-changed['history'][i]['val']['L_M'] for i in window]))
    window_total=float(np.mean([base['history'][i]['val']['total']-changed['history'][i]['val']['total'] for i in window]))
    # Registered development gate, not a family verdict or a sealed test rule.
    passed=mark_gain>=1 and total_gain>=1 and window_mark>=1 and window_total>=1
    result=dict(status='completed',tag=a.tag,battle='B4 development',stage='mark_memory_development_decision',
                development_gate=passed,selected_mark_loss_reduction=mark_gain,selected_total_loss_reduction=total_gain,
                common_window_epochs=window,common_window_mark_loss_reduction=window_mark,common_window_total_loss_reduction=window_total,
                rule='At least1 nat/sequence improvement in marks and total, at selected checkpoints and mean last20 common epochs; zero skipped updates.',
                selected_mark_memory=selected[0],selected_repaired_baseline=selected[1],
                accounting=[dict(arm=label,epochs=r['epochs_run'],train_targets=r['completed_train_targets'],
                                fitting_wall_s=r['fitting_wall_s'],seconds_per_train_target=r['fitting_wall_s']/r['completed_train_targets'],
                                evaluation_wall_s=r['evaluation_wall_s'],inference=r['inference'],max_rss_kb=r['max_rss_kb'])
                            for label,r in zip(('mark_memory','repaired_baseline'),records)],
                decision='Mark-memory gate passes: replicate on another DEV split/seed before selecting a developed protocol.' if passed else
                         'This mark variant fails its development gate: inspect time/mark and window differences before another change; no family-loss verdict.',
                scope='Fixed Github split1 seed0 TRAIN/DEV development, original core settings. Extra mark capacity/work measured separately; no matched-compute, published-reference or TEST win.',
                inputs={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
                source_sha256={'experiments/tpp/summarize_marks_development.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out=ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
