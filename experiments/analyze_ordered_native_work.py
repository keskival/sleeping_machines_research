"""Same-learning native-work comparison; CPU speed is not an ASIC claim."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();inputs={};sources={};records=[]
    for tag in ('aws_native_work_scalar_probe_20261013_v1','aws_ordered_native_work_probe_20261013_v1'):
        path=ROOT/f'experiments/results/token_language/{tag}.json';blob=path.read_bytes();r=json.loads(blob);assert r['status']=='completed'
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest();sources.update(r['source_sha256']);records.append(r)
    for name,digest in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def canon(r):return {k:v for k,v in r.items() if k not in ('wall_s','memory_effective_rank')}
    assert [canon(r) for r in records[0]['curve']]==[canon(r) for r in records[1]['curve']]
    assert records[0]['presentations_total']==records[1]['presentations_total']==8192
    rows=[]
    fit=('factual_core','alternative_core','training_readout','backward','optimizer','global_clip')
    for name,r in zip(('original_lookup','ordered_lookup'),records):
        w=r['native_work_accounting'];b=w['logical_payload_by_phase']['backward']
        rows.append(dict(implementation=name,targets=r['presentations_total'],registered_fit_flops_subtotal=w['registered_fit_flops'],
            declared_fit_scalar_arithmetic=sum(w['scalar_algorithm_work_by_phase'].get(phase,{}).get('scalar_arithmetic',0) for phase in fit),
            backward_scalar_arithmetic=w['scalar_algorithm_work_by_phase']['backward'].get('scalar_arithmetic',0),
            backward_operand_payload_bytes=b['input_tensor_payload_bytes'],backward_output_payload_bytes=b['output_tensor_payload_bytes'],
            optimizer_storage=w['optimizer_storage'],global_boundaries=w['synchronization'],
            operation_coverage_by_phase=w['operation_coverage_by_phase']))
    before,after=rows
    differences={name:before[name]-after[name] for name in ('registered_fit_flops_subtotal','declared_fit_scalar_arithmetic','backward_scalar_arithmetic','backward_operand_payload_bytes','backward_output_payload_bytes')}
    source=str(Path(__file__).relative_to(ROOT));sources[source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',exact_recorded_learning=True,rows=rows,removed_work=differences,
        input_sha256=inputs,source_sha256=sources,
        scope='Same8192-target integrated numerical trajectory/learner fingerprints, two instrumented implementations. Known scalar algorithms and registered matrix FLOPs have separate coverage and unknown operators remain explicit. Logical operand/output tensor exposures are not DRAM/ASIC wire traffic. Global clip/window backprop/dense Adam retained, same optimizer storage. Probe values are not extrapolated measured8M totals. Changed execution work is not a learned-credit replacement or silicon energy measurement.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(differences),flush=True)


if __name__=='__main__':main()
