"""B10 source-bound admission gates. Fits require its completed admission receipt."""
import argparse,copy,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
import torch
from b10_matched_data import prepare
from b10_matched_models import MatchedRace
from b10_operator_ledger import Ledger
ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    names=['experiments/market/b10_matched_pipeline.py','experiments/market/b10_matched_models.py','experiments/market/b10_matched_data.py','experiments/market/b10_operator_ledger.py','experiments/market/b10_depth_tpp_v1.py','experiments/tpp/race_tpp_v5.py','experiments/market/prepare.py','experiments/market/B10_AWS_MATCHED_CONTROLS.md']
    return {p:sha(ROOT/p) for p in names}
def update(model,opt,t,m):
    model.train();opt.zero_grad(set_to_none=True);loss=-model.terms(t,m).mean()
    assert torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
    opt.step();return float(loss.detach())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['gate','synthetic'],default='gate');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    start=time.monotonic();cpu=time.process_time();output=ROOT/f'experiments/results/market/{a.tag}.json';assert not output.exists()
    record=dict(status='running',battle='B10',source_sha256=sources(),training_admitted=False,rows=[],gates={})
    try:
        if a.mode=='synthetic':
            generator=np.random.default_rng(41);times=np.cumsum(generator.exponential(.01,(16,1024)),axis=1);times[:,3]=times[:,2]
            data={'train':[(t-t[0],generator.integers(0,6,1024)) for t in times]};receipt={}
        else:
            data,receipt=prepare();record['data_receipt_sha256']=sha(ROOT/'experiments/results/market/aws_b10_matched_data_20261014_v1.json')
        record['gates']['train_dev_window_counts']=a.mode!='synthetic'
        gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0]
        t=torch.tensor(np.stack([x[0] for x in data['train'][:16]]));m=torch.tensor(np.stack([x[1] for x in data['train'][:16]]))
        for family in ('native','lstm','transformer'):
            torch.manual_seed(0);model=MatchedRace(gaps,family);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4,foreach=False)
            model.eval();short=t[:1,:16].clone();marks=m[:1,:16].clone();short[:,3]=short[:,2]
            base=model.terms(short,marks);future_t=short.clone();future_m=marks.clone()
            future_t[:,9:]+=100.;future_m[:,9:]=(future_m[:,9:]+1)%6
            altered=model.terms(future_t,future_m)
            assert torch.equal(base[:,:8],altered[:,:8]),family+' causality'
            g1=torch.autograd.grad(base[:,:8].sum(),tuple(model.parameters()),allow_unused=True)
            g2=torch.autograd.grad(altered[:,:8].sum(),tuple(model.parameters()),allow_unused=True)
            assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(g1,g2))
            ledger=Ledger()
            # Representative measured full-length TRAIN update, charged later.
            with ledger:loss=update(model,opt,t,m)
            # Exact optimizer/model/RNG restoration witness.
            state=copy.deepcopy(model.state_dict());moments=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state()
            update(model,opt,short,marks);expected=copy.deepcopy(model.state_dict())
            model.load_state_dict(state);opt.load_state_dict(moments);torch.set_rng_state(rng)
            update(model,opt,short,marks)
            assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items())
            record['rows'].append(dict(family=family,parameters=sum(p.numel() for p in model.parameters()),full_shape_loss=loss,ledger=ledger.record(),causal_scores_and_gradients=True,resume_parity=True))
        if a.mode=='synthetic':
            record['status']='completed';record['gates']['synthetic_encoder_ledger_resume']=True;return
        record['gates']['original_checkpoint_required']=False
        record['status']='completed'
        record['gate_outcome']='awaiting_independent_ledger_validation_and_budget_freeze'
        record['reason']='Fresh fits need no old checkpoint; arithmetic audit and numerical B still required'
    except Exception as error:
        record['status']='completed';record['gate_outcome']='blocked_admission';record['reason']=str(error)
    finally:
        record.update(wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
