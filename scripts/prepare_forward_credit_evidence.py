"""Investor/report packet from completed Curie component results; no model imports."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAGS=dict(fresh='curie_optimizer_transport_fresh_20261011_v1_measure',execution='curie_transport_execution_20261011_v1_measure',controller='curie_token_controller_transport_variance_20261011_v1_measure',calibration='curie_token_optimizer_transport_20261011_v1_measure',tails='curie_token_credit_tail_20261011_v2_measure')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parents={};results={}
    for name,tag in TAGS.items():
        path=ROOT/f'experiments/results/credit/{tag}.json';d=json.loads(path.read_text());assert d['status']=='completed'
        parents[str(path.relative_to(ROOT))]=sha(path)
        for p,digest in d['source_sha256'].items():assert sha(ROOT/p)==digest,p
        results[name]=d
    f=results['fresh'];e=results['execution'];v=results['controller']
    mse={r['arm']:r['mse'] for r in f['rows']};variance={r['arm']:r['mean_conditional_parameter_gradient_variance'] for r in v['rows']}
    metrics=dict(credit_error_static_reduction_pct=100*(1-mse['transport_diagonal']/mse['static_full']),credit_error_reference_reduction_pct=100*(1-mse['transport_diagonal']/mse['reference_full']),credit_pipeline_speedup=e['whole_pipeline_speedup'],credit_supported_work_reduction_pct=100*(1-e['supported_work_ratio']),credit_controller_variance_reduction_pct=100*(1-variance['transport_diagonal']/variance['static']))
    packet=dict(status='completed',date='2026-10-11',battle='R1',completed_evidence_sha256=parents,metrics=metrics,
      fresh=dict(prefixes=8,targets_per_prefix=48,context=64,route_count=65,mse=mse,gate=f['accuracy_gate'],comparisons=f['comparisons'],scope='Prospective fixed-rule credit calibration on eight new TRAIN-tail prefixes; separate smoke. One observed future per prefix, source fixed two-layer model/50,257 GPT-2 token classes and64-dimensional lexical learner.'),
      execution=dict(prefixes=4,repeats=3,summaries=e['summaries'],scope='Same common carrier/candidate acquisition; exact all-route independent learners vs approximate transported credit plus exact factual/reference losses and full residual. Approximate predictor, exact conditional mean. Supported arithmetic omits unsupported operations included in wall.'),
      controller=dict(parameters=v['controller_parameters'],prefixes=4,rows=v['rows'],max_mean_error=max(r['mean_gradient_error'] for r in v['per_prefix']),scope='Fixed connected controller, exact conditional action-noise covariance trace; no future-distribution/native-carrier variance or optimization-speed result.'),
      status_text='Working proper-token hard-route/online lexical learner prototype; prospective credit accuracy and measured acquisition economy. Early512-target controller fits have no confirmed held-out utility improvement.',
      next_gate='Better held-out prediction per total learning cost, then larger data/model and online application confirmation.',
      potential='If the advantage scales: economical continual adaptation, more capable local machines, broader access to specialized learning, and a sparse asynchronous hardware path. Conditional platform opportunity, not measured industry impact or energy savings.')
    block_path=ROOT/'experiments/results/credit/curie_block_transport_fresh_20261011_v1_measure.json'
    if block_path.exists():
        block=json.loads(block_path.read_text());assert block['status']=='completed'
        for name,digest in block['source_sha256'].items():assert sha(ROOT/name)==digest,name
        parents[str(block_path.relative_to(ROOT))]=sha(block_path)
        means={arm:sum(row['mse'] for row in block['per_prefix'] if row['arm']==arm)/8 for arm in ['transport_block','transport_diagonal']}
        metrics['credit_block_diagonal_reduction_pct']=100*(1-means['transport_block']/means['transport_diagonal'])
        packet['block']=dict(mse=means,accuracy_gate=block['accuracy_gate'],comparison=block['comparisons']['block_over_transport_diagonal'],basis=block['block_basis_work'],scope=block['scope'])
    update_path=ROOT/'experiments/results/credit/curie_online_geometric_update_20261011_v3_fit.json'
    if not update_path.exists():update_path=ROOT/'experiments/results/credit/curie_online_geometric_update_20261011_v2_fit.json'
    if update_path.exists():
        update=json.loads(update_path.read_text());assert update['status']=='completed'
        for name,digest in update['source_sha256'].items():assert sha(ROOT/name)==digest,name
        parents[str(update_path.relative_to(ROOT))]=sha(update_path)
        packet['online_update']=dict(tag=update['tag'],comparison=update['comparisons']['online_over_frozen'],utility_gate=update['utility_gate'],config=update['config'],rows=update['rows'],pretraining=update['pretraining'],whole_wall_s=update['whole_wall_s'],scope=update['scope'])
        packet['status_text']='Working native temporal/keyed inference and lexical learning prototype; conditional credit accuracy/acquisition economy demonstrated. Actual online update-function learning is reported separately with its completed episode-level utility gate; no full-native task-learning win.'
    native_runs=[]
    for seed in range(3):
        path=ROOT/f'experiments/results/credit/aws_native_future_update_reuse_20261011_s{seed}.json'
        if not path.exists():continue
        native=json.loads(path.read_text());assert native['status']=='completed'
        for name,digest in native['source_sha256'].items():assert sha(ROOT/name)==digest,name
        parents[str(path.relative_to(ROOT))]=sha(path)
        for depth in [2,4,8]:
            arms={row['arm']:row for row in native['rows'] if row['depth']==depth};baseline=arms['fixed_adam'];learned=arms['future_taught']
            native_runs.append(dict(seed=seed,depth=depth,dev_gain=baseline['history'][-1]['nll']-learned['history'][-1]['nll'],fixed_fit_wall_s=baseline['fit_wall_s'],learned_fit_wall_s=learned['fit_wall_s'],train_target_presentations=learned['train_target_presentations']))
    if len(native_runs)==9:
        packet['native_optimizer_learning']=dict(rows=native_runs,positive_cases=sum(r['dev_gain']>0 for r in native_runs),mean_gain_by_depth={str(depth):sum(r['dev_gain'] for r in native_runs if r['depth']==depth)/3 for depth in [2,4,8]},scope='Three initializations, whole native depth2/4/8 width16,1400TaxiTRAIN sequences cropped32events/three passes,128reused DEV sequences. Future-trained180parameter group-scale controller retains exact actor backprop. All nine quality comparisons improve; complete CPU learning time increases. No efficient learned-core-credit or hardware claim.')
        packet['status_text']='Future-trained native optimizer improves development quality in nine of nine seed/depth cases; acquired-policy reuse is the next complete-work test. Generated deep credit improves versus frozen on six width8 cases but does not confirm superiority to nativeAdamW.'
        packet['date']='2026-10-12'
    for tag in ['curie_native_optimized_credit_20261011_v1_fit','curie_native_optimized_credit_20261011_v1_s1_fit','curie_native_optimized_credit_20261011_v1_s2_fit','curie_native_head_only_20261011_v1_measure','curie_native_event_credit_20261012_v1_d32_fit','curie_native_event_credit_20261012_v1_d16_fit']:
        path=ROOT/f'experiments/results/credit/{tag}.json'
        if path.exists():
            record=json.loads(path.read_text());assert record['status']=='completed';parents[str(path.relative_to(ROOT))]=sha(path)
    packet['generated_credit_status']='Optimized width8 online/frozen finalDEV gain6/6, superiority to own-nativeAdamW unconfirmed. Width16 deep credit improves versus head-only, width32 harms. Event-conditioned query field repairs online-versus-frozen atwidth32 but remains worse than head-only and nativeAdamW. Learned amplitude/momentum comparison running; no pending score promoted.'
    heads_path=ROOT/'experiments/results/credit/curie_native_event_heads_20261012_v1_measure.json'
    if heads_path.exists():
        heads=json.loads(heads_path.read_text());assert heads['status']=='completed';parents[str(heads_path.relative_to(ROOT))]=sha(heads_path)
        packet['matched_event_deep_utility']=dict(rows=[{k:r[k] for k in ['variant','width','depth','head_only_final_dev_nll','online_final_dev_nll','online_gain_vs_matched_heads']} for r in heads['rows']],scope=heads['scope'])
        packet['generated_credit_status']='Actual matched acquired-core/head controls: new event optimizer loses7/8; previous cross-configuration depth8 deep-benefit interpretation corrected. Local variants stop; full1400TaxiTRAIN threepasses/depth8/16/32 fresh-actor quality/work test running. Dense internal writes and episode resets remain.'
    cohort=[]
    for seed in [99913,99914,99915]:
        path=ROOT/f'experiments/results/credit/curie_native_cohort_event_20261012_v1_s{seed}_fit.json'
        if not path.exists():continue
        record=json.loads(path.read_text());assert record['status']=='completed'
        for name,digest in record['source_sha256'].items():assert sha(ROOT/name)==digest,name
        parents[str(path.relative_to(ROOT))]=sha(path)
        cohort.append(dict(seed=seed,config=record['config'],comparisons=[{k:r[k] for k in ['depth','online_gain_vs_frozen','online_gain_vs_matched_heads','online_gain_vs_adam','matched_quality_wall_ratio']} for r in record['rows']],scope=record['scope']))
    if cohort:
        packet['native_cohort_learning']=cohort;packet['generated_credit_status']='Complete native cohort/depth stage finished; source-bound seed/depth quality and complete matched-quality CPU comparisons recorded, exact matched heads/frozen/own-nativeAdamW. No public TEST or hardware claim.'
    # Native shared-world credit is a separate task-learning result, not a
    # replacement for the language component's accuracy/work measurements.
    native_credit={}
    for name,relative in {
        'fresh_system_confirmation':'experiments/results/causal_induction/curie_world_credit_confirm_20261012_v1_s194_decision.json',
        'original_answer_diagnosis':'experiments/results/diagnostics/aws_world_answer_changes_20261012_v1.json',
        'fresh_answer_diagnosis':'experiments/results/diagnostics/curie_fresh_world_answers_20261012_v1.json',
        'depth_utility_contract':'experiments/results/causal_induction/curie_direct_route_depth_20261012_v1_contract.json',
    }.items():
        path=ROOT/relative
        if not path.exists():continue
        record=json.loads(path.read_text());assert record['status']=='completed'
        for group in ['source_sha256','parent_sha256']:
            for source,digest in record.get(group,{}).items():assert sha(ROOT/source)==digest,source
        parents[relative]=sha(path)
        if name=='fresh_system_confirmation':
            native_credit[name]=dict(seed=record['seed'],confirmation_gate=record['confirmation_gate'],quality_gains=record['quality_gains'],depth_quality_gains=record['depth_quality_gains'],rows=[{k:v for k,v in row.items() if k!='rows'} for row in record['rows']],scope=record['scope'])
        elif name in ('original_answer_diagnosis','fresh_answer_diagnosis'):
            native_credit[name]=dict(rows=[{k:v for k,v in row.items() if k!='cases'} for row in record['rows']],scope=record['scope'])
        else:
            native_credit[name]=dict(result=relative,scope='Numerical identities and conditional error bounds; no learned optimizer or neural-depth scaling result.')
    packet['native_shared_world_credit']=native_credit
    market_runs=[]
    for tag in ['curie_b10_native_ttt_20261012_v3_fit','curie_b10_exact_future_20261012_v1_fit','curie_b10_longer_native_20261012_v1_fit','curie_b10_native_optimizer_20261012_v1_fit','curie_b10_regularized_optimizer_20261012_v1_fit']:
        path=ROOT/f'experiments/results/market/{tag}.json'
        if not path.exists():continue
        record=json.loads(path.read_text());assert record['status']=='completed'
        for source,digest in record['source_sha256'].items():assert sha(ROOT/source)==digest,source
        parents[str(path.relative_to(ROOT))]=sha(path)
        market_runs.append(dict(tag=tag,rows=[{k:v for k,v in row.items() if k in ['arm','nll','targets','wall_s','audits','future_teachers','critic_parameter_max_change']} for row in record['rows']],online_gain=record['online_gain'],acquisition_wall_s=record['acquisition']['wall_s'],scope=record['scope'],zero_arm_interpretation='Zero predictor with residual audits, not local-only' if 'native_ttt' in tag else 'Genuine local-only'))
    packet['native_market_online_learning']=market_runs
    repeated=ROOT/'experiments/results/causal_induction/curie_world_credit_confirm_20261012_v1_s195_decision.json'
    if repeated.exists():
        record=json.loads(repeated.read_text());assert record['status']=='completed'
        for group in ['source_sha256','parent_sha256']:
            for source,digest in record[group].items():assert sha(ROOT/source)==digest,source
        parents[str(repeated.relative_to(ROOT))]=sha(repeated)
        native_credit['fresh_acquisition_repeat']=dict(seed=record['seed'],confirmation_gate=record['confirmation_gate'],quality_gains=record['quality_gains'],depth_quality_gains=record['depth_quality_gains'],scope=record['scope'])
    assert f['accuracy_gate'] and all(x>0 for x in f['comparisons']['transport_over_static_full']['per_prefix_gain'])
    (ROOT/'investment/forward_credit_evidence_20261011.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps(metrics))
if __name__=='__main__':main()
