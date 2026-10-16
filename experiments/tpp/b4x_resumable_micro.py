"""B4 native memory repair: row microbatches,full histories,exact pass recovery."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
import race_tpp_b4x2 as X


def accumulate(model,batch,cell,budget):
    t,m,mask=batch;rows=max(1,budget//t.shape[1]);total=int(mask[:,1:].sum())
    for start in range(0,len(t),rows):
        sl=slice(start,start+rows);length=int(mask[sl].sum(1).max())
        tl,ml,n,_=model.loglik(t[sl,:length],m[sl,:length],mask[sl,:length],cell)
        loss=-(tl+ml)/total
        if not bool(torch.isfinite(loss)):return False
        loss.backward()
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())


def micro_update(model,opt,batch,cell,repairs,epoch,bi,budget):
    opt.zero_grad();finite=accumulate(model,batch,cell,budget)
    norm=X.rep.clip(model) if finite else torch.tensor(float('nan'))
    if finite and bool(torch.isfinite(norm)):opt.step();return
    # Preserve the original finite-sequence repair law without ever rebuilding
    # the memory-failing full padded batch. Apply one normalized update.
    t,m,mask=batch;kept=None;events=0;dropped=[]
    for row in range(len(t)):
        length=int(mask[row].sum());opt.zero_grad()
        tl,ml,n,_=model.loglik(t[row:row+1,:length],m[row:row+1,:length],mask[row:row+1,:length],cell)
        loss=-(tl+ml);loss.backward()
        grads=[p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
        if not bool(torch.isfinite(loss)) or not all(g is None or bool(torch.isfinite(g).all()) for g in grads):dropped.append(row);continue
        if kept is None:kept=grads
        else:kept=[a+b if a is not None and b is not None else (a if b is None else b) for a,b in zip(kept,grads)]
        events+=int(n)
    opt.zero_grad()
    if kept is not None and events:
        for p,g in zip(model.parameters(),kept):p.grad=None if g is None else g/events
        if bool(torch.isfinite(X.rep.clip(model))):opt.step()
    repairs.append(dict(epoch=epoch,batch=bi,dropped=dropped,retained_events=events,repair='sequence-accumulated,no full-batch replay'))


def save(path,state):
    temporary=path.with_suffix('.writing');torch.save(state,temporary);temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--split',type=int,required=True);p.add_argument('--local-clip',action='store_true');p.add_argument('--event-budget',type=int,default=2048);p.add_argument('--max-wall-s',type=float,default=10000);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(0);np.random.seed(0);random.seed(0);rng=random.Random(0)
    model,train,K,rules=X.build('github_filtered',a.split,0,True);val,_=X.load('github_filtered',a.split,'val');opt=torch.optim.AdamW(model.parameters(),lr=X.CFG['lr'],weight_decay=0.0)
    prefix=ROOT/f'experiments/results/tpp_b4/{a.tag}';checkpoint=prefix.with_suffix('.progress.pt');best_path=prefix.with_suffix('.pt');result=prefix.with_suffix('.json');assert not result.exists()
    history=[];repairs=[];best=math.inf;best_epoch=-1;next_epoch=0;past=0.;caps=None;X.rep.CAPS=None;start=time.monotonic()
    if checkpoint.exists():
        s=torch.load(checkpoint,map_location='cpu',weights_only=False);assert s['args']==vars(a)
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);torch.set_rng_state(s['torch_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng']);rng.setstate(s['order_rng'])
        history,repairs,best,best_epoch,next_epoch,past,caps=[s[k] for k in ['history','repairs','best','best_epoch','next_epoch','wall_s','caps']]
    elif a.local_clip:
        norms=[];model.train()
        for bi,batch in enumerate(X.base.batches(train,X.CFG['batch'],True,random.Random(1))):
            if bi>=32:break
            opt.zero_grad();assert accumulate(model,batch,rules['recording_cell'],a.event_budget),'nonfinite local-cap calibration'
            norms.append([float(p.grad.norm()) if p.grad is not None else 0. for p in model.parameters()])
        array=np.asarray(norms);caps=np.maximum(3*np.median(array,axis=0),1e-6).tolist();opt.zero_grad()
    X.rep.CAPS=caps
    for epoch in range(next_epoch,200):
        model.train();e0=time.monotonic()
        for bi,batch in enumerate(X.base.batches(train,X.CFG['batch'],True,rng)):micro_update(model,opt,batch,rules['recording_cell'],repairs,epoch,bi,a.event_budget)
        v=X.per_sequence(model,val,batch=1);history.append(dict(epoch=epoch,val_total=v['total'],val_L_T=v['L_T'],val_L_M=v['L_M'],epoch_s=time.monotonic()-e0));print(json.dumps(history[-1]),flush=True)
        if math.isfinite(v['total']) and v['total']<best:best,best_epoch=v['total'],epoch;save(best_path,model.state_dict())
        save(checkpoint,dict(args=vars(a),model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),order_rng=rng.getstate(),history=history,repairs=repairs,best=best,best_epoch=best_epoch,next_epoch=epoch+1,wall_s=past+time.monotonic()-start,caps=caps))
        if epoch-best_epoch>=30 or past+time.monotonic()-start>a.max_wall_s:break
    model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=True))
    files=X.SOURCES+['experiments/tpp/b4x_resumable_micro.py']
    out=dict(status='completed',battle='B4',attempt='developed,memory-repaired native pointer'+(' localclip' if a.local_clip else ''),dataset='github_filtered',split=a.split,args=dict(vars(a),seed=0,pointer=True),config=X.CFG,K=K,**rules,parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,epochs_run=len(history),history=history,repaired_updates=len(repairs),repairs=repairs,wall_s=past+time.monotonic()-start,checkpoint=str(best_path.relative_to(ROOT)),val=X.per_sequence(model,val,batch=1),test='not scored;frozen five-split development selection later',source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},scope='Native row-microbatch implementation repair;full sequences retained. Dropout/jitter coupling differs from old full batch,same per-example law. No test-based tuning. Exact pass checkpoints include optimizer,RNG,caps and order.')
    with result.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':main()
