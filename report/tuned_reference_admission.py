"""Admit complete, predeclared P0-6 groups; never select on partial TEST results."""
import json
import math
from pathlib import Path
import runpy
import shlex

ROOT=Path(__file__).resolve().parents[1]


def load_group(group,cap,result_root=None):
    result_root=Path(result_root or ROOT/'experiments/results/aws_20260929')
    fit=10_000_000 if group in ('A','B') else 90_000_000
    queues=sorted((ROOT/'experiments/queue').glob(f'aws_tuned_ref_{fit//1_000_000}M_{group}_*_20261004T210000Z.txt'))
    required={'A':6,'B':4,'C':2,'D':2}[group]
    if len(queues)!=required:raise ValueError('Changed predeclared tuned group')
    parser=runpy.run_path(str(ROOT/'experiments/check_tuned_reference_budgets.py'))['parser_from_source'](ROOT/'experiments/e64_lm_baselines.py')
    estimate=runpy.run_path(str(ROOT/'experiments/lm_training_flops.py'))['estimate_training_flops']
    rows=[];sources=set()
    for queue in queues:
        line=[l for l in queue.read_text().splitlines() if l and not l.startswith('#')]
        if len(line)!=1:raise ValueError('One-job tuned queue required')
        tokens=shlex.split(line[0]);arguments=tokens[tokens.index('--')+1:]
        directory=result_root/tokens[0];provenance=directory/'provenance.json'
        if not provenance.exists():continue
        p=json.loads(provenance.read_text())
        if p['status']!='completed':continue
        if p['script']!='experiments/e64_lm_baselines.py' or p['arguments']!=arguments:
            raise ValueError('Tuned result does not match declared queue')
        outputs=[f for f in directory.glob('*.json') if f.name!='provenance.json']
        if len(outputs)!=1:raise ValueError('One completed tuned model result required')
        r=json.loads(outputs[0].read_text());cfg=vars(parser.parse_args(arguments));a=r['args']
        if any(a.get(k)!=v for k,v in cfg.items()):raise ValueError('Tuned result arguments differ')
        steps=int(a['passes']*a['D']/(a['batch_size']*a['ctx']))
        if r['steps']!=steps:raise ValueError('Tuned fitting presentation count differs')
        work=estimate(a,r['params'],steps)['total_training_flops']
        if not math.isclose(work,r['training_flops_estimate']['total_training_flops'],rel_tol=1e-12):
            raise ValueError('Saved/recomputed tuned work differs')
        if work>cap:raise ValueError('Completed tuned arm exceeds actual native budget')
        for key in ('best_valid_bpc','test_bpc'):
            if not isinstance(r.get(key),(int,float)) or not math.isfinite(r[key]):raise ValueError('Finite tuned validation/test metric required')
        if a['D']!=fit or a['ctx']!=256 or a['test']!=1_000_000 or a['valid']!=200_000:
            raise ValueError('Tuned data/window protocol differs')
        sources.add(p['source_sha256']);rows.append(dict(r,path=str(outputs[0]),whole=work))
    if len(sources)>1:raise ValueError('Tuned group mixes numerical producer versions')
    if len(rows)!=required:return dict(status='pending',completed=len(rows),required=required,selected=None)
    # Original E64 LSTMs score validation with carried state, unlike native/TF.
    # TEST-only rescoring cannot repair validation selection across architectures.
    if any(r['args']['model']=='lstm' for r in rows):
        return dict(status='context_alignment_required',completed=len(rows),required=required,selected=None)
    return dict(status='completed',completed=len(rows),required=required,
                selected=min(rows,key=lambda r:r['best_valid_bpc']))
