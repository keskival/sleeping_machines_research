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
    assert f['accuracy_gate'] and all(x>0 for x in f['comparisons']['transport_over_static_full']['per_prefix_gain'])
    (ROOT/'investment/forward_credit_evidence_20261011.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps(metrics))
if __name__=='__main__':main()
