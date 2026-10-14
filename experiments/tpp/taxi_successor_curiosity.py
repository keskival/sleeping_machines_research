"""B1 exploratory Taxi successor: preserved parent plus temporal residual depth.
Only TRAIN/DEV select checkpoints; revisited TEST after both arms are frozen.
"""
import argparse,copy,hashlib,json,math,random,sys,time,resource
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tpp'));sys.path.insert(0,str(ROOT/'experiments/credit'))
import race_tpp_v19 as native
from native_training_linearization import NativeTrainingLinearization

class Successor(native.RaceTPP):
    def encode(self,t,m,mask):
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale
        age=(t-t[:,:1])/self.scale
        h=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1))
        for layer in self.layers:h=layer(h,dt)
        self.refine.eval()
        if torch.is_grad_enabled():
            self.packet=NativeTrainingLinearization.capture(self.refine,h,dt,context='taxi',version=self.credit_version)
            residual=self.packet.forward_output
        else:residual=self.refine(h,dt)
        h=h+self.residual_out(residual)
        return h,self.marks(h,m,dt)

def restored(record,train):
    a=record['args'];gaps=np.concatenate([np.diff(t) for t,_ in train]);pos=gaps[gaps>0]
    qs=np.log(np.quantile(pos,np.linspace(.1,.9,a['n_lognormal']))).tolist()
    model=native.RaceTPP(record['K'],a['d'],a['modes'],a['layers'],a['n_exp'],a['n_lognormal'],a['dv'],a['dropout'],record['scale'],qs,a['floor_cell'],a['n_window'],None,a['state_modes'])
    model.load_state_dict(torch.load(ROOT/record['checkpoint'],map_location='cpu',weights_only=True));return model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--epochs',type=int,default=20);a=ap.parse_args()
    out=ROOT/f'experiments/results/tpp/{a.tag}.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    parentpath=ROOT/'experiments/results/tpp/b1_r26_taxi_v19_rule_s2.json'
    parent=json.loads(parentpath.read_text());train=native.load_split('taxi','train');dev=native.load_split('taxi','dev')
    torch.manual_seed(2);baseline=restored(parent,train);candidate=copy.deepcopy(baseline);candidate.__class__=Successor
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1002);candidate.refine=native.TemporalMemoryLayer(32,16,0.);candidate.residual_out=torch.nn.Linear(32,32,bias=False)
        torch.nn.init.zeros_(candidate.residual_out.weight)
    candidate.credit_version=0
    rng=random.Random(2);baseline.eval();candidate.eval()
    t,m,mask=next(native.batches(dev,8,False,rng));x=baseline.event_terms(t,m,mask);y=candidate.event_terms(t,m,mask)
    zero_error=max(float((u-v).abs().max()) for u,v in zip(x[:2],y[:2]));assert zero_error<1e-12
    initial=native.evaluate(baseline,dev,8,rng,predict=False);assert abs(initial['ll']-parent['dev']['ll'])<1e-10
    # Original-parameter gradients must agree at zero residual under paired stochastic TRAIN inputs.
    baseline.train();candidate.train();rngstate=torch.get_rng_state().clone()
    tl,ml,n,_=baseline.loglik(t,m,mask,parent['args']['dequant_target']);(-(tl+ml)/n).backward()
    torch.set_rng_state(rngstate);tl,ml,n,_=candidate.loglik(t,m,mask,parent['args']['dequant_target']);(-(tl+ml)/n).backward()
    grads=dict(candidate.named_parameters());gradient_error=max(float((p.grad-grads[name].grad).abs().max()) for name,p in baseline.named_parameters() if p.grad is not None)
    assert gradient_error<1e-10
    baseline.zero_grad();candidate.zero_grad();rows=[];selected=[]
    for label,model in [('continued_parent',baseline),('preserved_depth_successor',candidate)]:
        torch.manual_seed(2002);rng=random.Random(2002);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0.)
        best=initial['ll'];best_epoch=-1;best_state=copy.deepcopy(model.state_dict());history=[];tick=time.monotonic();targets=0;updates=0
        for epoch in range(a.epochs):
            model.train()
            for t,m,mask in native.batches(train,32,True,rng):
                if label=='preserved_depth_successor':model.credit_version=updates
                tl,ml,n,_=model.loglik(t,m,mask,parent['args']['dequant_target']);loss=-(tl+ml)/n
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1;targets+=int(n)
            dv=native.evaluate(model,dev,8,rng,predict=False);history.append(dict(epoch=epoch,dev_ll=dv['ll']))
            if dv['ll']>best:best=dv['ll'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            print(label,epoch,dv['ll'],flush=True)
        duration=time.monotonic()-tick;model.load_state_dict(best_state)
        path=out.with_name(a.tag+'_'+label+'.pt');torch.save(best_state,path)
        selected.append(model);rows.append(dict(arm=label,dev_ll=best,best_epoch=best_epoch,fit_targets=targets,updates=updates,incremental_training_s=duration,parent_training_s=parent['wall_s'],whole_training_s=parent['wall_s']+duration,parameters=sum(p.numel() for p in model.parameters()),history=history,checkpoint=str(path.relative_to(ROOT))))
    # Both model choices are irrevocably frozen before TEST is opened.
    test=native.load_split('taxi','test')
    for row,model in zip(rows,selected):row['test']=native.evaluate(model,test,8,random.Random(2))
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and p.suffix=='.py' and p.relative_to(ROOT).parts[0] in ('experiments','sleeping_machines'):sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    inputs=[parentpath,ROOT/parent['checkpoint'],ROOT/'experiments/results/tpp/b1_final_unified_v19_taxi_s2.json']+[ROOT/f'data/easytpp/taxi/{s}.json' for s in ('train','dev','test')]
    old=json.loads(inputs[2].read_text())['test'];result=dict(status='completed',battle='B1',tag=a.tag,original_test=old,reference_ll=.522,zero_prediction_error=zero_error,original_gradient_error=gradient_error,rows=rows,wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},scope='Exploratory revisit of known Taxi seed2 TEST; two fixed20epoch continuations from saved parent, same TRAIN/DEV/order/dequantization/optimizer. Selection includes unchanged parent and uses DEV only. Temporal residual depth and cached exact primal packet integrated. Long-horizon learned critic, token-specific execution and market-only action heads are not fitted or transferred. No independent confirmatory benchmark win.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
