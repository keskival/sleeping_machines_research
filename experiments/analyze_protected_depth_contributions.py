"""R1 frozen endpoint decomposition: correction value versus fitted parent path.

No fitting, tuning, public validation, parameter change or checkpoint required
from another host. Execute on AWS through its bounded slot scheduler.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
import protected_depth_local_gate as gate
from public_token_scoring import score
from sleeping_machines.protected_branch_attribution import without_branch_memory, risk_decomposition

PREFIX='experiments/results/token_language/aws_protected_depth_'


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);started=time.monotonic()
    paths={arm:ROOT/(PREFIX+arm+'_2m_20261014_s6_v1.json') for arm in ('parent','grown')}
    records={arm:json.loads(path.read_text()) for arm,path in paths.items()}
    for arm,r in records.items():
        assert r['status']=='completed' and r['arm']==arm and r['added_updates']==4096
        for name,digest in r['source_sha256'].items():assert gate.pilot.driver.sha(ROOT/name)==digest,name
    for key in ('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset'):
        assert records['parent']['scored'][key]==records['grown']['scored'][key],key
    model,opt,args,train,warm,warm_path=gate.restore_arm('grown');del opt,train,warm
    checkpoint=paths['grown'].with_suffix('.pt');ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['binding']==records['grown']['binding']
    model.load_state_dict(ck['model']);before=parameter_hash(model);model.eval()
    dev=gate.pilot.driver.load_tokens(args.dev_file);actual=model.readout.nll
    original=type(model).forward;original_helper=original.__globals__['token_features']
    rows=[]
    for mode in ('factual','erase_added_history'):
        for name,rng in ck['owned_rng'].items():getattr(model,name).set_state(rng)
        model.core.new_depth_generator.manual_seed(args.seed+100001)
        namespace=dict(original.__globals__)
        if mode!='factual':namespace['token_features']=without_branch_memory(original_helper)
        forward=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        forward.__kwdefaults__=original.__kwdefaults__
        sums=dict(factual=0.,without_correction=0.,targets=0)
        parent_digest=hashlib.sha256()
        def audited(values,targets):
            losses=actual(values,targets)
            parent=values[...,:model.readout.parent_width]
            base=model.readout.base.nll(parent,targets)
            sums['factual']+=float(losses.double().sum())
            sums['without_correction']+=float(base.double().sum());sums['targets']+=targets.numel()
            parent_digest.update(parent.detach().contiguous().numpy().tobytes())
            return losses
        model.readout.nll=audited;begin=time.monotonic()
        try:
            metrics=score(model,dev,args,SimpleNamespace(Model=SimpleNamespace(forward=forward)),
                          25165824,8192,1024,8)
        finally:model.readout.nll=actual
        assert sums['targets']==8192 and abs(sums['factual']-metrics['loss_sum'])<1e-8
        rows.append(dict(mode=mode,scored=metrics,without_correction_nll=sums['without_correction']/8192,
            parent_feature_sha256=parent_digest.hexdigest(),scoring_s=time.monotonic()-begin))
    assert abs(rows[0]['scored']['nll']-records['grown']['scored']['nll'])<1e-10,'Saved score replay failed'
    assert rows[0]['parent_feature_sha256']==rows[1]['parent_feature_sha256'],'Parent changed under private memory intervention'
    assert rows[0]['without_correction_nll']==rows[1]['without_correction_nll']
    assert before==parameter_hash(model),'Frozen parameters changed'
    d=risk_decomposition(records['parent']['scored']['nll'],rows[0]['scored']['nll'],rows[0]['without_correction_nll'])
    d['added_history_erasure_effect']=rows[1]['scored']['nll']-rows[0]['scored']['nll']
    if d['correction_effect']>=0:
        decision='Correction does not improve its own fitted parent path: diagnose output/branch learning before more depth.'
    elif d['fitted_parent_path_excess']>0:
        decision='Correction helps its own fitted path but that path trails continued parent: investigate interference and preserve useful acquired learning before more depth.'
    else:
        decision='Both components improve this cohort; compare full work and confirm before scaling.'
    sources={name:gate.pilot.driver.sha(ROOT/name) for name in [
        'experiments/analyze_protected_depth_contributions.py','sleeping_machines/protected_branch_attribution.py',
        'experiments/public_token_scoring.py',*records['grown']['source_sha256']]}
    result=dict(status='completed',battle='R1',rows=rows,decomposition=d,decision=decision,
        parameter_sha256=before,source_sha256=sources,
        input_sha256={str(path.relative_to(ROOT)):gate.pilot.driver.sha(path) for path in [*paths.values(),checkpoint]},
        process_wall_s=time.monotonic()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen same8192 reusedDEVtargets/1024history/noEOSreset, seed6. Two scored passes, no tuning. Removing correction is a readout intervention; erasing only added-stage old memory retains protected parent and clocks random tape, but branch routes/delays can change. Not retrained ablation, isolated training causality, persistent-memory capacity theorem, public benchmark or quality promotion. Parent-path excess includes joint optimization, caps/warmup/teacher interactions; no unique causal attribution.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(decomposition=d,decision=decision)),flush=True)

if __name__=='__main__':main()
