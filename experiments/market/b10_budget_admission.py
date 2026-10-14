"""Measure every model's actual full updates and freeze B10 fitting arithmetic."""
import argparse,copy,json,math,resource,time
from pathlib import Path
import numpy as np
import torch
from b10_matched_models import MatchedRace
from b10_matched_data import prepare
from b10_matched_pipeline import ROOT,sha
from b10_matched_update import update
from b10_depth_tpp_v1 import Race
from b10_operator_ledger_v3 import Ledger

def source_hashes():
    names=['experiments/market/b10_budget_admission.py','experiments/market/b10_matched_update.py','experiments/market/b10_matched_fit.py','experiments/market/b10_matched_models.py','experiments/market/b10_matched_data.py','experiments/market/b10_matched_pipeline.py','experiments/market/b10_operator_ledger_v3.py','experiments/market/b10_depth_tpp_v1.py','experiments/market/prepare.py','experiments/tpp/race_tpp_v5.py','experiments/market/B10_AWS_MATCHED_CONTROLS.md']
    return {p:sha(ROOT/p) for p in names}
def numerical_ledger_audit():
    # Independent closed-form dot-product + reduction witnesses.
    a=torch.ones(3,4);b=torch.ones(4,5)
    with Ledger() as l:c=a@b
    assert l.flops==3*5*(2*4-1)
    with Ledger() as l:c=a.sum()
    assert l.flops==11
    with Ledger() as l:c=a.mean()
    assert l.flops==12
    with Ledger() as l:c=a.exp()
    assert l.special==12 and l.flops==0
    # Backward must be observed, not estimated from a forward multiplier.
    a.requires_grad_(True);b.requires_grad_(True)
    with Ledger() as l:(a@b).sum().backward()
    assert l.flops==105+14+12*9+20*5,(l.flops,l.record())
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    start=time.monotonic();cpu=time.process_time();out=ROOT/f'experiments/results/market/{a.tag}.json';assert not out.exists()
    record=dict(status='completed',battle='B10',training_admitted=False,source_sha256=source_hashes(),rows=[],gates={})
    try:
        record['gates']['closed_form_ledger_witnesses']=numerical_ledger_audit()
        data,receipt=prepare();record['data_receipt_sha256']=sha(ROOT/'experiments/results/market/aws_b10_matched_data_20261014_v1.json')
        gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0]
        t=torch.tensor(np.stack([x[0] for x in data['train'][:16]]));m=torch.tensor(np.stack([x[1] for x in data['train'][:16]]))
        costs={};overheads={};wall={}
        for family in ('native','lstm','transformer'):
            family_start=time.monotonic();torch.manual_seed(0);model=MatchedRace(gaps,family);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4,foreach=False)
            model.eval();short=t[:1,:16].clone();marks=m[:1,:16].clone();short[:,3]=short[:,2]
            base=model.terms(short,marks);other_t=short.clone();other_m=marks.clone();other_t[:,9:]+=100;other_m[:,9:]=(other_m[:,9:]+1)%6
            alt=model.terms(other_t,other_m);assert torch.allclose(base[:,:8],alt[:,:8],rtol=0,atol=1e-12),family+' causal score contract'
            parameters=tuple(model.parameters());g1=torch.autograd.grad(base[:,:8].sum(),parameters,allow_unused=True);g2=torch.autograd.grad(alt[:,:8].sum(),parameters,allow_unused=True)
            assert all((x is None and y is None) or (x is not None and y is not None and torch.allclose(x,y,rtol=1e-10,atol=1e-12)) for x,y in zip(g1,g2)),family+' causal gradient contract'
            # Reuse original head equations and weights with identical features.
            reference=Race(gaps,32,8,2);reference.m.load_state_dict({k:v for k,v in model.m.state_dict().items() if k in reference.m.state_dict()},strict=family=='native') if family=='native' else None
            if family=='native':reference.eval();assert torch.equal(reference.terms(short,marks),model.terms(short,marks))
            entries=[];costs[family]={};overhead=0
            for batch in (16,1):
                for repeat in (0,1):
                    before=time.monotonic()
                    with Ledger() as ledger:loss=update(model,opt,t[:batch],m[:batch])
                    measured=ledger.record();overhead+=ledger.flops
                    costs[family][str(batch)]=max(costs[family].get(str(batch),0),ledger.flops)
                    entries.append(dict(batch=batch,repeat=repeat,loss=loss,wall_s=time.monotonic()-before,ledger=measured))
            state=copy.deepcopy(model.state_dict());moments=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
            with Ledger() as resume_l: update(model,opt,short,marks)
            expected=copy.deepcopy(model.state_dict());model.load_state_dict(state);opt.load_state_dict(moments);torch.set_rng_state(rng)
            with Ledger() as replay_l:update(model,opt,short,marks)
            overhead+=resume_l.flops+replay_l.flops
            assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
            overheads[family]=overhead;wall[family]=sum(e['wall_s'] for e in entries if e['batch']==16)/2
            record['rows'].append(dict(family=family,parameters=sum(p.numel() for p in model.parameters()),updates=entries,causality_and_gradients=True,resume_parity=True,family_wall_s=time.monotonic()-family_start))
        B=30*(50*costs['native']['16']+costs['native']['1'])
        # Charge the first real LSTM pilot; its discarded state is not hidden work.
        pilot=ROOT/'experiments/results/market/aws_b10_lstm_reference_admission_20261014_v1.json'
        if pilot.exists():overheads['lstm']+=costs['lstm']['16']
        # All engineering updates are charged to each family's seed budget. This
        # conservative repeated charge prevents any seed receiving free tuning.
        record.update(training_admitted=True,budget_arithmetic=B,update_arithmetic=costs,overhead_arithmetic=overheads,timeouts_s={f:math.ceil(2*B/costs[f]['16']*wall[f]+7200) for f in costs},accounting_scope='executed training arithmetic under frozen scalar-kernel conventions; specials/comparisons separate; preprocessing and all gate wall/CPU disclosed',seed_list=[0,1,2],lrs=[.001,.003],test_sealed=True)
        record['gates'].update(full_window_updates=True,finite_tied_gap_gradients=True,causal_scores_and_gradients=True,optimizer_resume=True,original_native_likelihood_parity=True,no_unknown_operators=True)
    except Exception as error:
        record['reason']=type(error).__name__+': '+str(error);record['gate_outcome']='admission_failed';record['training_admitted']=False
    record.update(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
