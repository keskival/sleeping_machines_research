"""Source-bound B1/R1 first-pass learning-rule development decision."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def analyze(paths):
    records=[json.loads((ROOT/p).read_text()) for p in paths]
    assert len(records)==3 and all(r['status']=='completed' for r in records)
    assert {r['args']['arm'] for r in records}=={'future','immediate','scalar'}
    assert len({r['model_sha256'] for r in records})==1
    assert len({r['training_data_sha256'] for r in records})==1
    settings=('seed','model_seed','data_seed','meta_steps','inner_steps','batch','dev_tasks')
    assert len({tuple(r['args'][k] for k in settings) for r in records})==1
    rows=[];comparison={}
    for record in records:
        arm=record['args']['arm'];result=record['evaluation']['assessment']
        learned=next(r for r in result if r['arm']=='learned')
        comparison[arm]=learned
        baseline=min((r for r in result if r['arm'] in ('sgd','adam')),key=lambda r:r['mean_query_nll'])
        rows.append(dict(training_objective=arm,query_nll=learned['mean_query_nll'],
                         baseline_arm=baseline['arm'],baseline_lr=baseline['lr'],baseline_nll=baseline['mean_query_nll'],
                         gap_to_baseline=baseline['mean_query_nll']-learned['mean_query_nll'],
                         backward_parameters=record['backward_parameters'],whole_meta_training_wall_s=record['meta_training_wall_s'],
                         whole_run_wall_s=record['wall_s'],assessment_adaptation_wall_s=learned['wall_s'],
                         baseline_assessment_adaptation_wall_s=baseline['wall_s'],
                         meta_inner_targets=record['work_counts']['inner_gradient_targets'],
                         meta_outer_targets=record['work_counts']['outer_targets'],
                         adaptation_gradient_targets=learned['adaptation_gradient_targets'],
                         assessed_targets=learned['assessed_targets'],max_rss_kb=record['max_rss_kb']))
    future=comparison['future'];row=next(r for r in rows if r['training_objective']=='future')
    gaps={arm:comparison[arm]['mean_query_nll']-future['mean_query_nll'] for arm in ('immediate','scalar')}
    individual={arm:[x-y for x,y in zip(comparison[arm]['paired_change'],future['paired_change'])] for arm in gaps}
    gate=row['gap_to_baseline']>=.02 and all(g>=.01 for g in gaps.values())
    gate=gate and all(sum(v>0 for v in differences)/len(differences)>=.75 for differences in individual.values())
    return dict(status='completed',rows=rows,future_gain_over_objective_controls=gaps,
                paired_control_gain_mean={arm:statistics.mean(v) for arm,v in individual.items()},development_gate=gate,
                decision='Three-learner-seed confirmation under the same task/model protocol' if gate else 'Development: diagnose adaptation trajectories and update geometry before changing the rule; no automatic scaling',
                scope='Single learner seed, fixed forward initialization, synthetic development task distribution; no benchmark or total-work efficiency claim',
                inputs={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--inputs',nargs='+',required=True);a=ap.parse_args()
    result=analyze(a.inputs);me=Path(__file__).resolve()
    result.update(tag=a.tag,battle='B1/R1 reciprocal-learning enabler',source_sha256={str(me.relative_to(ROOT)):hashlib.sha256(me.read_bytes()).hexdigest()})
    with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
