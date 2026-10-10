"""R1/B1 completed-evidence gate; no fitting, TEST, or automatic scale launch."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--manifest',required=True);a=ap.parse_args()
    out=ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    packet=json.loads((ROOT/a.manifest).read_text())
    jobs=[j for j in packet['jobs'] if '_pilot_s' in j['tag'] and j['driver'].endswith('future_route_critic_v2.py')]
    if len(jobs)!=3:raise ValueError('Exactly three registered pilot seeds required')
    rows=[];inputs={};cost=[]
    for job in jobs:
        r=json.loads((ROOT/job['result']).read_text())
        if r['status']!='completed' or r['tag']!=job['tag']:raise ValueError('Result identity differs')
        if any(r['source_sha256'].get(k)!=v for k,v in job['result_sources'].items()):raise ValueError('Source binding differs')
        if r['args']['mode']!='pilot':raise ValueError('Pilot required')
        inputs[job['result']]=digest(job['result']);m=r['metrics'];ref=m['context_free_reference']
        for arm in m['arms']:
            d=arm['dev'];pair=d['versus_context_free']
            rows.append(dict(seed=r['args']['seed'],arm=arm['arm'],
                train_relative_mse=arm['train']['relative_mse'],dev_relative_mse=d['relative_mse'],
                constant_relative_mse=ref['dev_relative_mse'],
                dev_future_advantage=d['selected_future_advantage'],constant_future_advantage=ref['selected_future_advantage'],
                mse_reduction=pair['mean_squared_error_reduction'],loss_reduction=pair['mean_future_loss_reduction'],
                mse_ci=pair['mse_reduction_bootstrap95'],loss_ci=pair['future_loss_reduction_bootstrap95'],
                fitted=arm['train']['relative_mse']<1.,
                context_specific_point_signal=pair['mean_squared_error_reduction']>0 and pair['mean_future_loss_reduction']>0,
                context_specific_context_ci_signal=pair['mse_reduction_bootstrap95'][0]>0 and pair['future_loss_reduction_bootstrap95'][0]>0))
        cost.append(dict(seed=r['args']['seed'],whole_supported_flops=r['supported_flops'],
            teacher_scored_target_presentations=m['teacher_scored_targets'],wall_s=r['wall_s'],peak_rss_kb=r['peak_rss_kb']))
    if len({x['seed'] for x in rows})!=3:raise ValueError('Three distinct seeds required')
    verdicts=[]
    for arm in ('isolated','connected'):
        z=[x for x in rows if x['arm']==arm]
        if len(z)!=3:raise ValueError('Missing paired arm')
        replicated=all(x['context_specific_point_signal'] for x in z)
        if not all(x['fitted'] for x in z):decision='diagnose_fitting'
        elif not replicated:decision='diagnose_conditional_transfer_or_selection'
        else:decision='admit_design_of_coupled_future_learning_comparison'
        verdicts.append(dict(arm=arm,replicated_point_signal=replicated,
            all_context_intervals_positive=all(x['context_specific_context_ci_signal'] for x in z),
            mean_mse_reduction=statistics.mean(x['mse_reduction'] for x in z),
            mean_loss_reduction=statistics.mean(x['loss_reduction'] for x in z),decision=decision))
    record=dict(status='completed',tag=a.tag,battle='R1/B1',training=False,
        decision='Select the next development diagnosis from the registered conditional-credit gate; never declare an efficiency win',
        rows=rows,verdicts=verdicts,work=cost,input_sha256=inputs,
        source_sha256={'experiments/credit/summarize_future_credit.py':digest('experiments/credit/summarize_future_credit.py')},
        manifest_sha256=digest(a.manifest),
        scope='Frozen initialized native teachers; state interventions. Positive point improvements must repeat across all three seeds. Context bootstrap is conditional on each teacher/model, not a simultaneous or seed-level guarantee. Passing advances to coupled design, not automatic deep training.',
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out)
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
