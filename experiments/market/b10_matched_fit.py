"""Three-family B10 fits: charged two-LR selection and exact resumable budgets."""
import argparse,copy,json,math,resource,time
from pathlib import Path
import numpy as np
import torch
from b10_matched_data import prepare
from b10_matched_models import MatchedRace
from b10_matched_pipeline import sources,sha,ROOT
from b10_matched_update import update
from b10_operator_ledger_v3 import Ledger

@torch.no_grad()
def evaluate(model,windows):
    model.eval();ledger=Ledger();groups={}
    with ledger:
        for start in range(0,len(windows),4):
            rows=windows[start:start+4]
            t=torch.tensor(np.stack([r[0] for r in rows]));m=torch.tensor(np.stack([r[1] for r in rows]))
            tl,ml=model.component_terms(t,m);tl=tl[:,127:];ml=ml[:,127:]
            for i in range(len(rows)):
                day=str(windows[start+i][2]) if len(windows[start+i])>2 else str((start+i)//108)
                selections={'all':torch.ones_like(ml[i],dtype=torch.bool),'day:'+day:torch.ones_like(ml[i],dtype=torch.bool),
                    'tied':torch.diff(t[i])[127:]==0,'positive':torch.diff(t[i])[127:]>0}
                selections.update({'mark:'+str(k):m[i,128:]==k for k in range(6)})
                for key,mask in selections.items():
                    g=groups.setdefault(key,dict(targets=0,time_ll=0.,mark_ll=0.))
                    g['targets']+=int(mask.sum());g['time_ll']+=float(tl[i][mask].sum());g['mark_ll']+=float(ml[i][mask].sum())
    for g in groups.values():g['nll']=-(g['time_ll']+g['mark_ll'])/g['targets'] if g['targets'] else None
    return dict(nll=groups['all']['nll'],groups=groups,ledger=ledger.record())

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--family',choices=['native','lstm','transformer'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--admission',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    admission=json.loads((ROOT/a.admission).read_text())
    assert admission['status']=='completed' and admission['training_admitted']
    assert all(sha(ROOT/name)==digest for name,digest in admission['source_sha256'].items())
    data,receipt=prepare();assert admission['data_receipt_sha256']==sha(ROOT/'experiments/results/market/aws_b10_train_dev_sources_20261014_v1.json')
    gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0]
    # Attach actual calendar days, avoiding a uniform windows-per-day assumption.
    dev=[(t,m,identity['day']) for (t,m),identity in zip(data['dev'],receipt['identities']['dev'])]
    out=ROOT/f'experiments/results/market/{a.tag}.json';checkpoint=out.with_suffix('.progress.pt');best_path=out.with_suffix('.best.pt')
    assert not out.exists();start=time.monotonic();cpu=time.process_time()
    B=admission['budget_arithmetic'];costs=admission['update_arithmetic'][a.family]
    def new(lr):
        torch.manual_seed(a.seed);model=MatchedRace(gaps,a.family)
        return model,torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4,foreach=False)
    s=dict(phase=0,spent=admission['overhead_arithmetic'][a.family],phase_spent=0,epoch=0,cursor=0,order=[],targets=0,evals=[],candidates=[],best_nll=math.inf,wall_s=0.,cpu_s=0.,next_eval=0)
    model,opt=new(.001);nrng=np.random.default_rng(a.seed)
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False);assert saved['admission_sha256']==sha(ROOT/a.admission)
        assert saved['source_sha256']==admission['source_sha256'];s=saved['state'];model,opt=new(saved['lr'])
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);nrng.bit_generator.state=saved['numpy_rng']
    timing=[start,cpu]
    def save():
        s['wall_s']+=time.monotonic()-timing[0];s['cpu_s']+=time.process_time()-timing[1];timing[:]=[time.monotonic(),time.process_time()]
        tmp=checkpoint.with_suffix('.tmp');torch.save(dict(state=s,model=model.state_dict(),optimizer=opt.state_dict(),lr=opt.param_groups[0]['lr'],rng=torch.get_rng_state(),numpy_rng=nrng.bit_generator.state,admission_sha256=sha(ROOT/a.admission),source_sha256=admission['source_sha256']),tmp);tmp.replace(checkpoint)
    def score():
        result=evaluate(model,dev);s['evals'].append(dict(phase=s['phase'],spent=s['spent'],result=result))
        if result['nll']<s['best_nll']:
            s['best_nll']=result['nll'];torch.save(dict(model=model.state_dict(),nll=result['nll'],source_sha256=admission['source_sha256']),best_path)
        return result
    # Candidate boundaries are fixed fractions of B. Every smoke, initialization
    # and discarded candidate is charged through the frozen overhead ledger.
    while s['phase']<3:
        limit=math.floor(B*(.1 if s['phase']<2 else .8))
        if not s['order'] or s['cursor']>=len(s['order']):
            s['order']=nrng.permutation(len(data['train'])).tolist();s['cursor']=0;s['epoch']+=1
        indices=s['order'][s['cursor']:s['cursor']+16];batch=len(indices);cost=costs[str(batch)]
        if s['phase_spent']+cost>limit or s['spent']+cost>B:
            r=score()
            if s['phase']<2:
                candidate=out.with_suffix(f'.candidate{s["phase"]}.pt')
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),numpy_rng=nrng.bit_generator.state,order=s['order'],cursor=s['cursor'],lr=opt.param_groups[0]['lr']),candidate)
                s['candidates'].append(dict(nll=r['nll'],path=str(candidate.relative_to(ROOT))))
            s['phase']+=1;s['phase_spent']=0;s['next_eval']=0
            if s['phase']==1:
                model,opt=new(.003);nrng=np.random.default_rng(a.seed);s['order']=[];s['cursor']=0
            elif s['phase']==2:
                selected=min(s['candidates'],key=lambda x:x['nll']);c=torch.load(ROOT/selected['path'],weights_only=False)
                model,opt=new(c['lr']);model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng']);nrng.bit_generator.state=c['numpy_rng'];s['order']=c['order'];s['cursor']=c['cursor']
            save();continue
        rows=[data['train'][i] for i in indices];t=torch.tensor(np.stack([r[0] for r in rows]));m=torch.tensor(np.stack([r[1] for r in rows]))
        # Frozen upper-bound costs include tie-sensitive kernels; actual ledger
        # must not exceed the certified bound. No post-result budget change.
        with Ledger() as executed:update(model,opt,t,m)
        assert executed.flops<=cost,(executed.flops,cost)
        s['spent']+=cost;s['phase_spent']+=cost;s['targets']+=batch*1023;s['cursor']+=batch
        if s['phase']==2 and s['phase_spent']>=limit*(s['next_eval']+1)/4:
            score();s['next_eval']+=1
        if s['cursor']>=len(s['order']):save()
    slack=(B-s['spent'])/B
    assert 0<=slack<=.01,('budget slack exceeds frozen protocol',slack)
    best=torch.load(best_path,weights_only=False);model.load_state_dict(best['model']);final=evaluate(model,dev)
    result=dict(status='completed',battle='B10',family=a.family,seed=a.seed,source_sha256=admission['source_sha256'],admission_sha256=sha(ROOT/a.admission),budget=B,fitting_arithmetic=s['spent'],unused_fraction=slack,targets=s['targets'],evaluations=s['evals'],final_dev=final,final_weights=str(best_path.relative_to(ROOT)),wall_s=s['wall_s']+time.monotonic()-timing[0],cpu_s=s['cpu_s']+time.process_time()-timing[1],max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,september_opened=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
