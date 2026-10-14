"""R1: preserve completed acquisition, then extend its trajectory to64M."""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import allocated_depth_token_language as driver
from sleeping_machines.acquisition_budget_extension import extend_acquisition_checkpoint
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host

PARENT='curie_allocated_deep_d8_32m_20261013_s6_v1'
TAG='curie_allocated_deep_d8_64m_20261014_s6_v1'
CONSEQUENCE='curie_crosswindow_d8_8m_s6_20261013_v1_resume'


def main():
    deadline=time.monotonic()+14*86400
    print('Waiting for completed deep ladder and cross-window comparison; no training slot occupied',flush=True)
    while subprocess.run(['tmux','has-session','-t','curie_crosswindow_language_20261013_v1'],capture_output=True).returncode==0:
        if time.monotonic()>deadline:raise RuntimeError('Prior pipeline exceeded14day wait envelope')
        time.sleep(20)
    decision=json.loads((ROOT/f'experiments/results/diagnostics/{CONSEQUENCE}_decision.json').read_text())
    assert decision['status']=='completed'
    preparation_started=time.monotonic()
    packet=original.load(PARENT);original.load(PARENT+'_quality','diagnostics')
    parent=ROOT/f'experiments/results/token_language/{PARENT}.pt'
    for name,digest in packet['source_sha256'].items():
        assert driver.sha(ROOT/name)==digest,name
    checkpoint=driver.torch.load(parent,map_location='cpu',weights_only=False)
    for field,filename in [('train_sha256',packet['args']['train_file']),('dev_sha256',packet['args']['dev_file'])]:
        assert driver.sha(ROOT/filename)==checkpoint['identity'][field],filename
    steps=131072
    candidate=extend_acquisition_checkpoint(checkpoint,packet,steps)
    before=dict(checkpoint);before.pop('settings')
    after=dict(candidate);after.pop('settings')
    assert driver.learner_fingerprint(before)==driver.learner_fingerprint(after)
    settings=dict(candidate['settings']);settings['steps']=checkpoint['settings']['steps']
    assert settings==checkpoint['settings']
    args=packet['args'];data=driver.load_tokens(ROOT/args['train_file'])
    reader=driver.BlockLaneTokens(data,args['train_tokens'],args['lanes'],'fresh',steps,args['chunk'])
    assert reader.admitted<=len(data)
    rate=packet['train_wall_s_total']/packet['presentations_total']
    additional=(steps-checkpoint['step'])*args['lanes']*args['chunk']
    timeout=math.ceil(2*rate*additional+3600)
    if timeout>604800:raise RuntimeError('Measured incremental acquisition exceeds7days; optimize execution')
    base=ROOT/f'experiments/results/token_language/{TAG}_migration'
    if base.with_suffix('.pt').exists() or base.with_suffix('.json').exists():
        raise FileExistsError('Never overwrite a prior migration')
    driver.torch.save(candidate,base.with_suffix('.pt'))
    receipt=dict(status='prepared',battle='R1',parent=PARENT,child=TAG,
        parent_checkpoint_sha256=driver.sha(parent),migration_checkpoint_sha256=driver.sha(base.with_suffix('.pt')),
        retained_trajectory_fingerprint=driver.learner_fingerprint(before),
        previous_steps=checkpoint['step'],steps=steps,unique_admitted_train_tokens=reader.admitted,
        available_train_tokens=len(data),migration_wall_s=time.monotonic()-preparation_started,
        source_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in
            (Path(__file__),ROOT/'sleeping_machines/acquisition_budget_extension.py')},
        scope='Explicit budget-only migration, not an ordinary same-settings resume. Actor/Adam/numerical state/audits/allRNG/cursor/curves preserved; endpoint and future data admission alone extended. No new quality evidence.')
    with base.with_suffix('.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    del checkpoint,candidate,before,after,data,reader
    stable_host();original.run(TAG,max(timeout,5400));result=original.load(TAG)
    result['acquisition_budget_extension']=receipt
    result['whole_fit_process_wall_s']=packet.get('whole_fit_process_wall_s',packet['process_wall_s'])+result['process_wall_s']+receipt['migration_wall_s']
    result['whole_fit_work_scope']='Parent complete process plus child complete process and measured migration validation/serialization preparation. Successful-trajectory timer carries parent acquisition once. Parent exact work required.'
    if 'whole_fit_process_wall_lower_bound_s' in packet:
        result['whole_fit_process_wall_lower_bound_s']=packet['whole_fit_process_wall_lower_bound_s']+result['process_wall_s']+receipt['migration_wall_s']
        result.pop('whole_fit_process_wall_s')
        result['whole_fit_work_scope']='Parent lower-bound process work plus child and migration; unknown interruption tail retained.'
    output=ROOT/f'experiments/results/token_language/{TAG}.json';tmp=output.with_suffix('.work.tmp')
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(output)
    original.run(TAG+'_quality',5400);quality=original.load(TAG+'_quality','diagnostics')
    paths=[str(output.relative_to(ROOT)),str(base.with_suffix('.json').relative_to(ROOT)),
           f'experiments/results/diagnostics/{TAG}_quality.json']
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Complete native64M acquisition preserving32M trajectory'],cwd=ROOT,check=True)
    print(json.dumps({'status':'completed','selected_long_history_dev':quality['selected_long_history_dev']}),flush=True)


if __name__=='__main__':main()
