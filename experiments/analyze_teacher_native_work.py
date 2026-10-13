"""R1 admitted-teacher work ledger with independent numerical replay witness."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):
 if isinstance(v,dict):return {k:canon(x) for k,x in v.items() if k not in ('wall_s','memory_effective_rank')}
 if isinstance(v,list):return [canon(x) for x in v]
 return v

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();sources={};inputs={};records=[]
 for tag in ('aws_ordered_native_work_probe_20261013_v1','aws_teacher_native_reference_20261013_v1','aws_teacher_native_probe_20261013_v1'):
  path=ROOT/f'experiments/results/token_language/{tag}.json';r=json.loads(path.read_text());assert r['status']=='completed';inputs[str(path.relative_to(ROOT))]=sha(path);sources.update(r['source_sha256']);records.append(r)
 for name,digest in sources.items():assert sha(ROOT/name)==digest,name
 baseline,reference,probe=records
 assert canon(reference['curve'])==canon(probe['curve']), 'Instrumentation changed learning'
 assert baseline['presentations_total']==reference['presentations_total']==probe['presentations_total']==8192
 for key in baseline['args']:
  if key not in ('tag','resume','stop_after_step'):assert baseline['args'][key]==probe['args'][key],key
 fit=('factual_core','alternative_core','training_readout','backward','optimizer','global_clip');rows=[]
 for label,r in (('full_teacher',baseline),('admitted_teacher',probe)):
  w=r['native_work_accounting'];phases=w['scalar_algorithm_work_by_phase']
  rows.append(dict(model=label,targets=8192,final_dev_nll=r['curve'][-1]['dev_nll'],registered_fit_flops_subtotal=w['registered_fit_flops'],declared_fit_scalar_arithmetic=sum(phases.get(k,{}).get('scalar_arithmetic',0) for k in fit),logical_fit_input_bytes=sum(w['logical_payload_by_phase'].get(k,{}).get('input_tensor_payload_bytes',0) for k in fit),logical_fit_output_bytes=sum(w['logical_payload_by_phase'].get(k,{}).get('output_tensor_payload_bytes',0) for k in fit),setup_and_unclassified_scalar_work=phases.get('other_unclassified',{}),semantic_events=w['semantic_events_by_phase'],update_boundaries=w['synchronization'],optimizer_storage=w['optimizer_storage'],operation_coverage=w['operation_coverage_by_phase'],teacher_work=r.get('teacher_work')))
 before,after=rows;removed={key:before[key]-after[key] for key in ('registered_fit_flops_subtotal','declared_fit_scalar_arithmetic','logical_fit_input_bytes','logical_fit_output_bytes')}
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 d=dict(status='completed',battle='R1',instrumentation_preserves_learning=True,rows=rows,removed_work=removed,input_sha256=inputs,source_sha256=sources,scope='Same8192-target acquisition/settings, full vs probability0.25 teacher. Different learner trajectories and quality reported, no iso-quality work win inferred from this probe. Independent admitted uninstrumented fit must match instrumented learning exactly except nondifferentiable SVD rounding. Registered matrix/scalar counts have named partial coverage; unclassified/setup work and unsupported operators retained. Logical tensor exposures are not measured traffic/energy; global clipping, window backward and denseAdam remain. Main quality evidence comes from completed three-seed128k/512k cohorts; no8M extrapolation.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(removed),flush=True)
if __name__=='__main__':main()
