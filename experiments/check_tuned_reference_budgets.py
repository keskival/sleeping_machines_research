"""P0-6 queue budget audit, standard library only; never launch a queue."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import shlex

ROOT=Path(__file__).resolve().parents[1]


def parser_from_source(path):
    parser=argparse.ArgumentParser()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument':
            kw={k.arg: ({'int':int,'float':float,'str':str}[k.value.id] if k.arg=='type'
                       else ast.literal_eval(k.value)) for k in n.keywords}
            parser.add_argument(*(ast.literal_eval(a) for a in n.args),**kw)
    return parser


def audit():
    estimate=runpy.run_path(str(ROOT/'experiments/lm_training_flops.py'))['estimate_training_flops']
    parser=parser_from_source(ROOT/'experiments/e64_lm_baselines.py')
    native=runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))
    data=native['load'](lambda p:json.loads((native['RES']/p).read_text()))
    a=next(r for r in data['native'] if r['label']=='p96/d4 + route credit, 6 passes')
    b=next(r for r in data['native'] if r['label']=='p64/d4 + route credit, 4 passes')
    c=next(r for r in data['native90'] if r['label']=='p64/d4/pool2 + route credit')
    projected_positions=int(4*(90_000_000-1)//8192)*8192
    budgets={'A':a['whole'],'B':b['whole'],'C':c['fit']*projected_positions,
             'D':a['fit']*projected_positions}
    rows=[]
    for queue in sorted((ROOT/'experiments/queue').glob('aws_tuned_ref_*_20261004T210000Z.txt')):
        lines=[l for l in queue.read_text().splitlines() if l and not l.startswith('#')]
        if len(lines)!=1: raise ValueError('One-job queue required')
        tokens=shlex.split(lines[0]);cfg=vars(parser.parse_args(tokens[tokens.index('--')+1:]))
        group=queue.name.split('_')[4];width=cfg['size'];layers=cfg['layers']
        params=(layers*(12*width**2+13*width)+(2*27+cfg['ctx'])*width+27 if cfg['model']=='tf'
                else 4*width*(width+64)+35*width+27*64+27)
        steps=int(cfg['passes']*cfg['D']/(cfg['batch_size']*cfg['ctx']))
        work=estimate(cfg,params,steps)['total_training_flops']
        if work>budgets[group]: raise ValueError('Queued arm exceeds native budget: '+queue.name)
        rows.append(dict(queue=str(queue.relative_to(ROOT)),queue_sha256=hashlib.sha256(queue.read_bytes()).hexdigest(),
                         group=group,parameters=params,updates=steps,whole_fit_flops=work,
                         native_budget_flops=budgets[group],fraction=work/budgets[group],
                         budget_status='within budget',context='continuous state' if cfg['model']=='lstm' else 'reset T256 windows',
                         protocol_status='context alignment required' if cfg['model']=='lstm' else 'aligned native window geometry'))
    assert len(rows)==14
    return dict(status='completed',queues_checked=len(rows),within_budget=len(rows),
                lstm_queues_requiring_context_alignment=sum(r['protocol_status']=='context alignment required' for r in rows),
                rows=rows,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    ('experiments/check_tuned_reference_budgets.py','experiments/e64_lm_baselines.py',
                     'experiments/lm_training_flops.py','report/native_language_batched_appendix.py')},
                scope='A/B budgets from completed native fits; C/D projected from saved per-position traces and queued fitting presentations. '
                      'Actual C/D budgets must be rechecked after completion. Existing LSTM scoring carries state; same-window validation '
                      'and test rescoring is required before an identical-context tuned claim. No queues or numerical work launched.',
                native_forward_calls=0,backward_calls=0,optimizer_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=ROOT/args.output
    if output.exists():raise ValueError('Preserve prior result')
    result=audit();output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','queues_checked','within_budget','lstm_queues_requiring_context_alignment')}))
