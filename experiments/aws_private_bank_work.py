"""Complete guarded arithmetic replay of one completed private-bank fit."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))


def assert_same(left,right,path='checkpoint'):
    """Compare numeric training state, including route/teacher generators."""
    import torch
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):
            raise ValueError('Replay tensor mismatch: '+path)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('Replay mapping mismatch: '+path)
        for k in left:assert_same(left[k],right[k],path+'.'+str(k))
    elif isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('Replay sequence mismatch: '+path)
        for i,(x,y) in enumerate(zip(left,right)):assert_same(x,y,path+'.'+str(i))
    elif type(left)!=type(right) or left!=right:raise ValueError('Replay scalar mismatch: '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--control-tag',required=True);a=p.parse_args()
    packet=json.loads(Path(a.manifest).read_text());pins=packet['work_source_sha256']
    def check():
        for name,digest in pins.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source mismatch: '+name)
    check()
    if a.control_tag not in packet['fit_tags']:raise ValueError('Undeclared control')
    folder=ROOT/'experiments/results/token_language';control_path=folder/(a.control_tag+'.json')
    control=json.loads(control_path.read_text())
    if control.get('status')!='completed' or control.get('source_sha256')!=packet['fit_source_sha256']:
        raise ValueError('Completed source-bound bank control required')
    if not control.get('bank_recipe'):raise ValueError('Private-bank recipe missing')
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    pending=out.with_suffix('.pending.json')
    if out.exists() or pending.exists():raise FileExistsError('Unique work receipt required')
    import torch
    import token_stage_work as work
    from aws_private_bank_model import PrivateBankModel
    parent=work.engine.Model;old_argv=sys.argv
    work.engine.Model=PrivateBankModel
    sys.argv=['token_stage_work.py','--control',str(control_path),'--tag',a.tag,'--output',str(pending)]
    try:work.main()
    finally:work.engine.Model=parent;sys.argv=old_argv
    check();record=json.loads(pending.read_text())
    original=torch.load(folder/(a.control_tag+'.pt'),weights_only=False)
    replay=torch.load(folder/(a.tag+'.pt'),weights_only=False)
    fields=('model','optimizer','state','cursor','step','site_generator','position_generator',
            'local_generator','alternative_generator','generator','torch_rng','writes','total_presentations')
    for field in fields:assert_same(original[field],replay[field],field)
    for section in ('fitting','inference'):
        if record[section]['formula_coverage_complete'] is not True:
            raise ValueError('Incomplete work formula coverage: '+section)
    record['underlying_work_source_sha256']=record['source_sha256']
    record['source_sha256']=pins
    record['bank_recipe']=control['bank_recipe']
    record['final_numeric_training_state_exact']=True
    record['exact_checkpoint_fields']=list(fields)
    record['max_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['ordinary_fit_targets_per_second']=control['train_tokens_per_second']
    record['ordinary_fit_max_rss_kb']=control['max_rss_kb']
    out.write_text(json.dumps(record,indent=2)+'\n')
    pending.unlink()


if __name__=='__main__':main()
