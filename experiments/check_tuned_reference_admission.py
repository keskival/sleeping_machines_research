"""P0-6 complete-group/provenance regression checks, without numerical imports."""
import argparse
import builtins
import copy
import json
from pathlib import Path
import runpy
import shlex
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def check():
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('torch','numpy','h5py','matplotlib'):raise AssertionError('No numerical import: '+name)
        return original(name,*a,**kw)
    builtins.__import__=guarded
    try:
        api=runpy.run_path(str(ROOT/'report/tuned_reference_admission.py'))['load_group']
        parser=runpy.run_path(str(ROOT/'experiments/check_tuned_reference_budgets.py'))['parser_from_source'](ROOT/'experiments/e64_lm_baselines.py')
        estimate=runpy.run_path(str(ROOT/'experiments/lm_training_flops.py'))['estimate_training_flops']
        definitions=[]
        for q in sorted((ROOT/'experiments/queue').glob('aws_tuned_ref_10M_A_*_20261004T210000Z.txt')):
            line=next(l for l in q.read_text().splitlines() if l and not l.startswith('#'));tok=shlex.split(line)
            arguments=tok[tok.index('--')+1:];a=vars(parser.parse_args(arguments));w=a['size']
            params=(a['layers']*(12*w*w+13*w)+(54+a['ctx'])*w+27 if a['model']=='tf'
                    else 4*w*(w+64)+35*w+1755)
            steps=int(a['passes']*a['D']/(a['batch_size']*a['ctx']))
            definitions.append((tok[0],dict(status='completed',script='experiments/e64_lm_baselines.py',arguments=arguments,source_sha256='a'*64),
                                dict(args=a,params=params,steps=steps,best_valid_bpc=2.,test_bpc=1.,
                                     training_flops_estimate=estimate(a,params,steps))))
        passed=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);assert api('A',352.1e12,root)['completed']==0
            for i,(tag,prov,row) in enumerate(definitions):
                directory=root/tag;directory.mkdir();(directory/'provenance.json').write_text(json.dumps(prov))
                (directory/'model.json').write_text(json.dumps(row))
                result=api('A',352.1e12,root)
                assert result['completed']==i+1 and result['selected'] is None
                assert result['status']==('pending' if i<5 else 'context_alignment_required')
            passed+=['empty group pending','each partial group pending; no premature validation selection',
                     'complete original LSTM/TF group withheld until validation/test contexts align']
            tag,prov,row=definitions[-1];path=root/tag/'model.json';provenance=root/tag/'provenance.json'
            for mutation in ('arguments','work','updates','nan_validation','nan_test','data','source','queue'):
                bad=copy.deepcopy(row);p=copy.deepcopy(prov)
                if mutation=='arguments':bad['args']['lr']=.123
                elif mutation=='work':bad['training_flops_estimate']['total_training_flops']*=.5
                elif mutation=='updates':bad['steps']-=1
                elif mutation=='nan_validation':bad['best_valid_bpc']=float('nan')
                elif mutation=='nan_test':bad['test_bpc']=float('nan')
                elif mutation=='data':bad['args']['test']=999999
                elif mutation=='source':p['source_sha256']='b'*64
                else:p['arguments'].append('--changed')
                path.write_text(json.dumps(bad));provenance.write_text(json.dumps(p))
                try:api('A',352.1e12,root)
                except ValueError:pass
                else:raise AssertionError('Invalid completed arm admitted: '+mutation)
                path.write_text(json.dumps(row));provenance.write_text(json.dumps(prov))
            passed.append('eight completed-arm provenance/metric/count/source/data mutations rejected')
            try:api('A',1,root)
            except ValueError:pass
            else:raise AssertionError('Over-budget tuned arm admitted')
            passed.append('actual native budget enforced without rounded-cap allowance')
        return dict(status='completed',checks=passed,checks_passed=len(passed),
                    numerical_imports=0,forward_calls=0,backward_calls=0,optimizer_updates=0,
                    scope='Synthetic JSON/queue publication fixtures only; no trained tuned reference or new quality score.')
    finally:builtins.__import__=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=ROOT/args.output
    if output.exists():raise ValueError('Unique output required')
    result=check();output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
