"""Bounded mixed-type integration witness; ONLY run via queue/run_safe.sh."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sleeping_machines.typed_predicate_interface import Field,Predicate
from sleeping_machines.typed_temporal_model import TypedTemporalModel


def predicates():
    return [Predicate('positive',Field('temperature','numeric')),
        Predicate('wood',Field('material','categorical',('oak','steel')),members=frozenset({'oak'})),
        Predicate('enabled',Field('enabled','boolean'))]


def rows(count,seed):
    rng=random.Random(seed); data=[]; labels=[]
    for _ in range(count):
        x=rng.uniform(-1,1);x=None if rng.random()<.15 else x
        material=rng.choice(('oak','steel')); enabled=bool(rng.randrange(2))
        data.append(dict(temperature=x,material=material,enabled=enabled))
        labels.append(int(x is None or ((x>0) ^ (material=='oak') ^ enabled)))
    return data,torch.tensor(labels,dtype=torch.long)


def contracts(model):
    data,_=rows(8,9102);ids=model.encode(data)
    shuffled=[dict(reversed(list(row.items()))) for row in data]
    assert torch.equal(ids,model.encode(shuffled))
    recoded=copy.deepcopy(model)
    recoded.predicates=[Predicate(p.name,Field('material','categorical',('z','a')),members=frozenset({'z'})) if p.name=='wood' else p for p in model.predicates]
    relabeled=[dict(row,material='z' if row['material']=='oak' else 'a') for row in data]
    assert torch.equal(ids,recoded.encode(relabeled))
    probe=dict(data[0],temperature=0.);missing=dict(probe,temperature=None)
    assert not torch.equal(model.encode([probe]),model.encode([missing]))
    generator=torch.Generator().manual_seed(7)
    with torch.no_grad():
        model.eval(); a,state,_=model(ids,generator,deterministic=True)
        model(model.encode(shuffled[::-1]),generator,deterministic=True)
        model.train(); b,_,_=model(ids,generator,deterministic=True)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        c,_,_=model(ids[:1],generator,deterministic=True)
        torch.testing.assert_close(a[:1],c,rtol=1e-5,atol=1e-6)
    loss,audit=model.fitting_loss(ids,torch.arange(8)%2,
        torch.Generator().manual_seed(8),torch.Generator().manual_seed(9),torch.Generator().manual_seed(10))
    loss.backward()
    key_grad=model.core.banks['key'].grad
    assert key_grad is not None and bool(torch.isfinite(key_grad).all()) and float(key_grad.abs().sum())>0
    assert any(abs(x)>0 for x in audit['terminal_loss_difference'])
    model.zero_grad(set_to_none=True)
    return dict(column_presentation_invariant=True,category_relabel_invariant=True,
        missing_distinct_from_zero=True,independent_row_reset=True,
        training_inference_same_deterministic_path=True,finite_nonzero_key_credit=True,
        actual_alternative_write_changes_terminal_risk=True,
        observed_updates=int(state['last_writes'].sum()),teacher=audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--steps',type=int,default=64);p.add_argument('--seed',type=int,default=6);p.add_argument('--save-weights',action='store_true')
    a=p.parse_args();out=ROOT/'experiments/results/typed_tabular'/f'{a.tag}.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(a.seed);start=time.monotonic()
    model=TypedTemporalModel(predicates());proof=contracts(model)
    fit,targets=rows(64,22001);dev,dev_y=rows(256,22002);ids=model.encode(fit);dev_ids=model.encode(dev)
    generator=torch.Generator().manual_seed(a.seed+1);sites=torch.Generator().manual_seed(a.seed+2);alternatives=torch.Generator().manual_seed(a.seed+3)
    opt=torch.optim.AdamW(model.parameters(),lr=.003);curve=[];audit=None
    def evaluate(step):
        with torch.no_grad():
            logits,_,_=model(dev_ids,torch.Generator().manual_seed(918),deterministic=True)
            row=dict(step=step,dev_nll=float(torch.nn.functional.cross_entropy(logits,dev_y)),dev_accuracy=float((logits.argmax(-1)==dev_y).float().mean()))
            curve.append(row);print(json.dumps(row),flush=True)
    evaluate(0)
    for step in range(1,a.steps+1):
        begin=((step-1)*16)%len(fit);opt.zero_grad(set_to_none=True)
        loss,audit=model.fitting_loss(ids[begin:begin+16],targets[begin:begin+16],generator,sites,alternatives)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite integrated objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite integrated gradient')
        opt.step()
        if step%16==0 or step==a.steps:evaluate(step)
    sources=['sleeping_machines/typed_temporal_model.py','sleeping_machines/typed_predicate_interface.py','sleeping_machines/sparse_counterfactual_episodes.py','sleeping_machines/sparse_counterfactual_layer.py','sleeping_machines/packed_token_core.py','sleeping_machines/paired_route_credit.py',str(Path(__file__).relative_to(ROOT))]
    result=dict(status='completed',args=vars(a),contracts=proof,curve=curve,last_teacher=audit,fit_rows=64,dev_rows=256,fitting_row_presentations=a.steps*16,parameters=sum(p.numel() for p in model.parameters()),wall_s=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Synthetic mixed-type interaction witness, one seed; fixed predicates and canonical complete comparison presentation. Existing integrated temporal/sparse core, learned receiver selection and actual alternative-write terminal-risk credit. No learned thresholds/predicate discovery, tree benchmark, scaling/transfer or complete FLOP claim.')
    out.parent.mkdir(parents=True,exist_ok=True)
    if a.save_weights:
        weights=out.with_suffix('.pt')
        if weights.exists():raise FileExistsError(weights)
        torch.save(model.state_dict(),weights)
        result['final_weights']=str(weights.relative_to(ROOT))
        result['final_weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
