"""Immutable completed-measurement input for the planned scaling plots/fits.

Stdlib only. This collector never fits a curve or executes a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    rows=[];provenance={};family=None
    for budget,label in ((65536,'64k'),(262144,'256k'),(1048576,'1m')):
        for width in (16,24,32):
            for seed in (6,7):
                tag=f'curie_data_growth_tokens_{label}_b64_c16_p{width}_s{seed}_20261005_v1'
                if (budget,width,seed)==(1048576,24,7):
                    tag='curie_original_1m_p24_s7_20261006_v1'
                path=ROOT/f'experiments/results/token_language/{tag}.json'
                selection_path=path.with_suffix('.selection.json')
                if not path.exists() or not selection_path.exists():continue
                result=json.loads(path.read_text());selection=json.loads(selection_path.read_text())
                if result['status']!='completed' or selection['status']!='completed':continue
                a=result['args']
                assert (a['train_tokens'],a['payload'],a['seed'])==(budget,width,seed)
                per_pass=a['lanes']*(budget//a['lanes']-1)
                assert result['presentations_total']==2*per_pass
                assert a['steps']==2*((budget//a['lanes']-1+a['chunk']-1)//a['chunk'])
                expected=[0,*range(a['eval_every'],a['steps']+1,a['eval_every'])]
                assert [x['step'] for x in result['curve']]==expected and len(expected)==5
                settings={k:v for k,v in a.items() if k not in ('tag','payload','seed','train_tokens','steps','eval_every')}
                identity=(settings,result['identity'])
                if family is None:family=identity
                assert family==identity,'Changed model/data/protocol family'
                chosen=selection['selected']
                assert abs(chosen['dev_nll']-min(x['dev_nll'] for x in result['curve']))<2e-6
                row=dict(tag=tag,seed=seed,payload=width,admitted_train_tokens=budget,
                         fitting_targets=result['presentations_total'],passes=2,
                         initial_nll=result['curve'][0]['dev_nll'],selected_nll=chosen['dev_nll'],
                         final_nll=result['curve'][-1]['dev_nll'],selected_step=chosen['step'],
                         parameters=result['parameters'],core_and_input_parameters=result['core_parameters'],
                         readout_parameters=result['readout_parameters'],tail_widths=result['decoder_tail_widths'],
                         available_receivers=a['depth']*a['heads']*a['pool'],
                         addressed_memory_scalars_per_lane=a['depth']*a['heads']*a['pool']*width,
                         selected_writes_per_target=a['depth']*a['heads'],
                         scored_keys_per_target=a['depth']*a['heads']*a['pool'],
                         measured_targets_per_second=result['train_tokens_per_second'],rss_kib=result['max_rss_kb'],
                         complete_work=None)
                provenance[str(path.relative_to(ROOT))]=digest(path)
                provenance[str(selection_path.relative_to(ROOT))]=digest(selection_path)
                work=ROOT/f'experiments/results/diagnostics/curie_data_growth_tokens_{label}_p{width}_work_20261005_v1.json'
                # Current work queue audits seed6; never lend its cost to a seed7 fit.
                if seed==6 and work.exists():
                    audit=json.loads(work.read_text())
                    assert audit['status']=='completed' and Path(audit['control']).name==path.name
                    assert audit['fitting_targets']==row['fitting_targets'] and audit['curve_parity_max_error']<2e-6
                    assert all(audit[k]['formula_coverage_complete'] for k in ('fitting','inference'))
                    row['complete_work']={k:audit[k] for k in ('fitting','inference','fitting_arithmetic_flops_per_target','inference_arithmetic_flops_per_target')}
                    provenance[str(work.relative_to(ROOT))]=digest(work)
                rows.append(row)
    primary={(r['admitted_train_tokens'],r['payload']) for r in rows if r['seed']==6}
    required={(d,w) for d in (65536,262144,1048576) for w in (16,24,32)}
    return dict(status='completed_measurement_collection',rows=rows,input_sha256=provenance,
                missing_primary_cells=[list(x) for x in sorted(required-primary)],
                crossed_quality_packet_complete=required<=primary,
                curve_fitted=False,producer_source_sha256=digest(Path(__file__)),
                scope='Completed development measurements only; no inferred law. Width changes core/input and decoder jointly; fixed selected write count is not fixed FLOPs. All two-pass work remains charged even when selected checkpoint is earlier. Cost axis uses only actual complete arithmetic audits; no projections or seed6 costs assigned to seed7. 4M remains reserved for extrapolation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    record=collect();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(rows=len(record['rows']),crossed_quality_packet_complete=record['crossed_quality_packet_complete'],curve_fitted=False)))
