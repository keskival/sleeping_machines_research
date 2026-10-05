"""Completed-only width selection within the preserved64K integrated recipe."""
import argparse
import hashlib
import json
from pathlib import Path

from aws_token_stage_evidence import stage_row, add_utility
ROOT=Path(__file__).resolve().parents[1]


def decide(records,selections,utilities):
    widths=set(records)
    if widths not in ({16,24},{16,24,32}) or set(selections)!=widths or set(utilities)!=widths:
        raise ValueError('Completed P16/P24 pair, optionally P32, required with selections/utilities')
    baseline=None;rows={}
    for width in sorted(widths):
        result=records[width];args=result['args']
        if args['payload']!=width or args['train_tokens']!=65536 or args['seed']!=6:
            raise ValueError('Cell differs from declared64K seed6 width study')
        settings={k:v for k,v in args.items() if k not in ('tag','payload')}
        identity=(settings,result['identity'])
        if baseline is None:baseline=identity
        elif identity!=baseline:raise ValueError('Width study data/source/optimizer/history protocol differs')
        row=stage_row(result,selections[width]);add_utility(row,utilities[width])
        if row['equivalent_passes']!=2 or row['fitting_targets']!=131056:
            raise ValueError('Width study requires same two-pass target exposure')
        rows[width]=row
    learning=[w for w,r in rows.items() if r['selected_step']>0 and r['learning_gain']>=.02 and r['context_gain']>0]
    selected=min(learning,key=lambda w:(rows[w]['selected_dev_nll'],rows[w]['parameters'])) if learning else None
    baseline_loss=rows[16]['selected_dev_nll']
    comparisons={str(w):dict(quality_gain_against_p16=baseline_loss-r['selected_dev_nll'],
        parameter_ratio_against_p16=r['parameters']/rows[16]['parameters'],
        state_scalar_ratio_against_p16=r['persistent_memory_scalars_per_lane']/rows[16]['persistent_memory_scalars_per_lane'],
        scored_keys_unchanged=r['scored_keys_per_target']==rows[16]['scored_keys_per_target'],
        selected_writes_unchanged=r['selected_writes_per_target']==rows[16]['selected_writes_per_target'],
        verdict=('single-seed development accuracy win' if r['selected_dev_nll']<baseline_loss
                 else 'single-seed development accuracy loss' if r['selected_dev_nll']>baseline_loss else 'tie'))
        for w,r in rows.items() if w!=16}
    return dict(status='completed',completed_widths=sorted(widths),rows=rows,
        comparisons=comparisons,selected_width=selected,
        selection_rule='Lowest initial-inclusive DEV NLL among completed members with >=.02 learning gain and positive contextual utility; ties choose fewer parameters.',
        next_action='Repeat selected width in seed7 and audit complete work before larger data' if selected else 'Diagnose completed learning trajectories; no scaling admission',
        independent_seed_confirmed=False,scaling_ready=False,
        required_next_evidence=['same selected width at seed7','actual complete fitting/inference ledger',
                                'reserved256K quality/capacity point','reserved larger extrapolation test'],
        scope='Same64K data,131056presentations,optimizer/evaluation cadence,seed6 and causal history. Width changes core and default decoder ranks together. More vector width changes active arithmetic even at equal selected write count. This is not a fixed-activity dormant-bank test, matched-FLOP win or a scaling exponent. Message/memory erasures remain frozen interventions, not retrained architecture selection.')


def collect(root):
    records={};selections={};utilities={};inputs={};pending=[]
    for width in (16,24,32):
        tag=f'curie_data_growth_tokens_64k_b64_c16_p{width}_s6_20261005_v1'
        base=root/'experiments/results/token_language';p=base/(tag+'.json');s=base/(tag+'.selection.json')
        utility_name=('curie_data_growth_tokens_64k_utility_20261005_v1' if width==16
                      else f'curie_data_growth_tokens_64k_p{width}_utility_20261005_v1')
        u=root/'experiments/results/diagnostics'/(utility_name+'.json')
        if not all(x.exists() for x in (p,s,u)):
            pending.append(width);continue
        result,selection,utility=[json.loads(x.read_text()) for x in (p,s,u)]
        if any(d.get('status')!='completed' for d in (result,selection,utility)):
            pending.append(width);continue
        matching=[r for r in utility['rows'] if r['tag']==tag]
        if len(matching)!=1:raise ValueError('Utility tag binding failed')
        records[width]=result;selections[width]=selection;utilities[width]=matching[0]
        for path in (p,s,u):inputs[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if 16 not in records or 24 not in records:
        return dict(status='awaiting_completed_width_pair',completed_widths=sorted(records),pending_widths=pending,
                    selected_width=None,scaling_ready=False,input_sha256=inputs)
    decision=decide(records,selections,utilities)
    decision.update(pending_widths=pending,input_sha256=inputs,
                    producer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return decision


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=ROOT/a.output
    if out.exists():raise FileExistsError(out)
    decision=collect(ROOT);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps({k:v for k,v in decision.items() if k not in ('rows','input_sha256')}))


if __name__=='__main__':main()
