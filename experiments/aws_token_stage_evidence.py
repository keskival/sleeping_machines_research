"""Source/result-bound common quality/work rows for token scaling stages.

Stdlib only. Completed trajectories supply quality; actual matching whole-fit
ledgers supply work. Missing ledgers remain None, never estimates or zero.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage_row(result, selection, work=None):
    if result.get('status') != 'completed' or selection.get('status') != 'completed':
        raise ValueError('Completed fit and selection required')
    args = result['args']; curve = result['curve']; selected = selection['selected']
    if not curve or curve[0]['step'] != 0 or any(not math.isfinite(r['dev_nll']) for r in curve):
        raise ValueError('Finite initial-inclusive trajectory required')
    expected=[0]+list(range(args['eval_every'],args['steps']+1,args['eval_every']))
    if expected[-1]!=args['steps']:expected.append(args['steps'])
    if [r['step'] for r in curve]!=expected:
        raise ValueError('Complete declared development selection schedule required')
    best = min(curve, key=lambda r: (r['dev_nll'], r['step']))
    if best['step'] != selected['step'] or not math.isclose(best['dev_nll'], selected['dev_nll'], abs_tol=1e-10, rel_tol=0.):
        raise ValueError('Selection must match best initial-inclusive curve')
    if Path(selection['original_result']).stem != args['tag']:
        raise ValueError('Selection belongs to another fit')
    train_admitted = args['train_tokens']//args['lanes']*args['lanes']
    targets_per_pass = train_admitted-args['lanes']
    dev_targets = args['dev_tokens']//args['eval_lanes']*args['eval_lanes']-args['eval_lanes']
    if min(targets_per_pass, dev_targets, result['presentations_total']) <= 0:
        raise ValueError('Positive actual target populations required')
    if args['dev_offset'] < 10485760:
        raise ValueError('Public validation must remain reserved')
    row = dict(tag=args['tag'], seed=args['seed'], payload=args['payload'],
        depth=args['depth'], heads=args['heads'], pool=args['pool'],
        credit_window=args['credit_window'], optimizer_chunk=args['chunk'],
        admitted_train_tokens=train_admitted, fitting_targets=result['presentations_total'],
        equivalent_passes=result['presentations_total']/targets_per_pass,
        dev_targets=dev_targets, dev_offset=args['dev_offset'],
        parameters=result['parameters'], core_parameters=result['core_parameters'],
        readout_parameters=result['readout_parameters'],
        persistent_memory_scalars_per_lane=args['depth']*args['heads']*args['pool']*args['payload'],
        scored_keys_per_target=args['depth']*args['heads']*args['pool'],
        selected_writes_per_target=args['depth']*args['heads'],
        initial_dev_nll=curve[0]['dev_nll'], selected_dev_nll=selected['dev_nll'],
        learning_gain=curve[0]['dev_nll']-selected['dev_nll'], selected_step=selected['step'],
        train_targets_per_s=result['train_tokens_per_second'], max_rss_kb=result['max_rss_kb'],
        whole_fit_gflops=None, fitting_mflops_per_target=None, inference_mflops_per_target=None,
        fitting_special_functions=None, inference_special_functions=None,
        data_identity={k:result['identity'][k] for k in ('train_sha256','dev_sha256')},
        evaluation_cadence=args['eval_every'], work_scope=None)
    if result['parameters'] != result['core_parameters']+result['readout_parameters']:
        raise ValueError('Parameter capacity split inconsistent')
    if work is None:
        return row
    if work.get('status') != 'completed' or Path(work['control']).stem != args['tag']:
        raise ValueError('Completed matching control ledger required')
    if work['fitting_targets'] != row['fitting_targets'] or work['inference_targets'] != dev_targets:
        raise ValueError('Work and quality target denominators differ')
    if work['optimizer_updates'] != args['steps'] or work['curve_parity_max_error'] >= 2e-6:
        raise ValueError('Whole-fit work must preserve the complete trajectory')
    if not math.isclose(work['selected_dev_nll'], selected['dev_nll'], abs_tol=2e-6, rel_tol=0.):
        raise ValueError('Inference work is for another selected quality')
    for label, count in (('fitting', row['fitting_targets']), ('inference', dev_targets)):
        ledger=work[label]
        if not ledger['formula_coverage_complete'] or ledger['unsupported_floating_operators']:
            raise ValueError('Incomplete arithmetic formula coverage')
        flops=ledger['arithmetic_flops']; specials=ledger['special_function_evaluations']
        if not math.isfinite(flops) or flops<=0 or not math.isfinite(specials) or specials<0:
            raise ValueError('Invalid executed work')
        reported=work[label+'_arithmetic_flops_per_target']
        if not math.isclose(reported, flops/count, rel_tol=1e-12):
            raise ValueError('Work normalization inconsistent')
        row[label+'_mflops_per_target']=flops/count/1e6
        row[label+'_special_functions']=specials
    row['whole_fit_gflops']=work['fitting']['arithmetic_flops']/1e9
    row['work_scope']=work['scope']
    return row


def add_utility(row, utility):
    if utility['tag'] != row['tag'] or utility['fit_targets'] != row['fitting_targets'] or utility['dev_targets'] != row['dev_targets']:
        raise ValueError('Utility populations differ from selected fit')
    selected=utility['selected']
    if selected['step'] != row['selected_step'] or not math.isclose(selected['dev_nll'], row['selected_dev_nll'], rel_tol=0., abs_tol=2e-6):
        raise ValueError('Utility belongs to another checkpoint selection')
    if utility.get('partition_parity') is not True or utility.get('matched_rng') is not True:
        raise ValueError('Utility needs matched route RNG/partition evidence')
    metrics={'context_gain':utility['context_gain'],
        'memory_erasure_cost':utility['history_gains']['memory'],
        'message_erasure_cost':utility['history_gains']['message']}
    if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite utility')
    row.update(metrics)
    row['utility_checkpoint_sha256']=utility['checkpoint_sha256']
    return row


def collect(root):
    folder=root/'experiments/results/token_language'
    rows=[]; inputs={}; pending=[]
    tags=[f'curie_fixed_batch_tokens_8k_b64_c{w}_s{s}_20261005_v1' for w in (16,64) for s in (6,7)]
    tags += [f'curie_data_growth_tokens_64k_b64_c16_p{p}_s6_20261005_v1' for p in (16,24,32)]
    for tag in tags:
        path=folder/(tag+'.json'); selection_path=folder/(tag+'.selection.json')
        if not path.exists() or not selection_path.exists():
            pending.append(tag); continue
        result=json.loads(path.read_text());selection=json.loads(selection_path.read_text())
        if result.get('status')!='completed' or selection.get('status')!='completed':
            pending.append(tag);continue
        work_path=root/'experiments/results/diagnostics/curie_fixed_batch_tokens_8k_c16_work_20261005_v1.json'
        work=None
        paths=[path,selection_path]
        if tag=='curie_fixed_batch_tokens_8k_b64_c16_s6_20261005_v1' and work_path.exists():
            work=json.loads(work_path.read_text());paths.append(work_path)
        row=stage_row(result,selection,work)
        row.update(context_gain=None,memory_erasure_cost=None,message_erasure_cost=None)
        utility_name=('curie_fixed_batch_tokens_8k_utility_20261005_v1' if tag.startswith('curie_fixed_batch_')
                      else 'curie_data_growth_tokens_64k_utility_20261005_v1' if result['args']['payload']==16
                      else f"curie_data_growth_tokens_64k_p{result['args']['payload']}_utility_20261005_v1")
        if utility_name:
            utility_path=root/'experiments/results/diagnostics'/(utility_name+'.json')
            if utility_path.exists():
                utility=json.loads(utility_path.read_text())
                if utility.get('status')=='completed':
                    matches=[r for r in utility['rows'] if r['tag']==tag]
                    if len(matches)!=1:raise ValueError('Missing or duplicate utility row')
                    add_utility(row,matches[0]);paths.append(utility_path)
        rows.append(row)
        for p in paths:inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    # This surface shares the development population; prior/cadence changes are
    # explicit row fields, not interpreted as an isolated learned gain.
    populations={(r['data_identity']['dev_sha256'],r['dev_offset'],r['dev_targets']) for r in rows}
    if len(populations)>1:raise ValueError('Different development populations require separate tables')
    return dict(status='completed',rows=rows,pending=pending,input_sha256=inputs,
        producer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Completed GPT-2 FineWeb native development stages on one scored population. Frequency priors/cadences can differ; absolute NLL and within-fit learning are separate. Arithmetic/special functions separate; sampling work unquantified and initialization/evaluation excluded from fit FLOPs. Missing whole-fit ledgers remain null. No public benchmark, scaling exponent or matched-reference win inferred.')


def markdown(record):
    text=['# Integrated token quality and complete work','',record['scope'],'',
        '| TRAIN tokens | P | Seed | Credit | Fit targets | Passes | State scalars/lane | Keys/writes per target | Initial NLL | Selected NLL | Learning gain | Whole-fit GFLOPs | Fit MFLOPs/target | Eval MFLOPs/target |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:'Pending' if value is None else f'{value:.6f}'
    for r in record['rows']:
        values=[r['admitted_train_tokens'],r['payload'],r['seed'],r['credit_window'],r['fitting_targets'],f"{r['equivalent_passes']:.2f}",r['persistent_memory_scalars_per_lane'],f"{r['scored_keys_per_target']}/{r['selected_writes_per_target']}"]
        values += [fmt(r[k]) for k in ('initial_dev_nll','selected_dev_nll','learning_gain','whole_fit_gflops','fitting_mflops_per_target','inference_mflops_per_target')]
        text.append('| '+' | '.join(map(str,values))+' |')
    text += ['', '| TRAIN tokens | P | Seed | Credit | Context gain NLL | Memory erasure cost NLL | Message erasure cost NLL |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for r in record['rows']:
        values=[r['admitted_train_tokens'],r['payload'],r['seed'],r['credit_window']]
        values += [fmt(r.get(k)) for k in ('context_gain','memory_erasure_cost','message_erasure_cost')]
        text.append('| '+' | '.join(map(str,values))+' |')
    text += ['', 'Frozen utility uses matched route RNG and selected weights; negative erasure cost means the intervention improves loss. Context gain uses a constant causal TRAIN-mean feature through the same readout. These are not retrained ablations or additive attribution.', '']
    text += ['', 'Pending stage names: ' +', '.join(record['pending'])+'.', '',
        'Available memory, key scoring, selected writes and learning work are distinct columns. Decoder and optimizer work are included in executed ledgers; state size is not a FLOP saving. Evaluation here includes exact likelihood/scorer reductions.', '']
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=ROOT/a.output
    if out.exists() or out.with_suffix('.md').exists():raise FileExistsError(out)
    record=collect(ROOT);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2)+'\n');out.with_suffix('.md').write_text(markdown(record))
    print(json.dumps(dict(rows=len(record['rows']),pending=record['pending'])))


if __name__=='__main__':main()
