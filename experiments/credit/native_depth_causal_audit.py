"""R1 paired causal depth interventions on acquired weights; no optimizer updates.

A layer bypass removes its computation, write and delay while consuming the same
race draws. Gain-zero removes only its residual proposal correction. Both are
explicit interventions at fixed D, not claims of exact shallow-model recovery.
"""
import argparse,json,sys,time,resource
from pathlib import Path
from types import FunctionType,SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import bernoulli_teacher_token_language as driver
import sleeping_machines.sparse_counterfactual_episodes as episodes
from sleeping_machines.ordered_token_episodes import ContentFunctions
from public_token_scoring import reference_forward

class StageAudit:
    def __init__(self,depth,mode='observe',target=None):
        self.depth=depth;self.mode=mode;self.target=target;self.index=0;self.records=[]
    def layer(self,*args,**kwargs):
        d=self.index%self.depth;self.index+=1;values=list(args)
        if self.mode=='gain_zero' and d==self.target:values[23]=0.
        result=episodes.sparse_counterfactual_layer(*values,**kwargs)
        # Native step arguments: x,arrival,mem,arr,seen,reads,...,gain(index23).
        if self.mode=='bypass' and d==self.target:
            x,arrival,mem,arr,seen,reads=values[:6];H=mem.shape[1];P=mem.shape[-1]
            result=(x,arrival,mem,arr,seen,reads,x.reshape(len(x),H,P),arrival[:,None].expand(-1,H),result[8],result[9])
        x,arrival=values[:2];y,delivery=result[:2]
        row=dict(stage=d,input=x,output=y,arrival=arrival,delivery=delivery)
        if x.requires_grad:x.retain_grad()
        if y.requires_grad:y.retain_grad()
        self.records.append(row);return result
    def token_features(self,*args,token_content,**kwargs):
        assert not kwargs.get('compiled',False)
        fn=episodes.token_features;namespace=dict(fn.__globals__)
        namespace['F']=ContentFunctions(args[1],token_content);namespace['sparse_counterfactual_layer']=self.layer
        patched=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__);patched.__kwdefaults__=fn.__kwdefaults__
        return patched(*args,**kwargs)
    def forward(self,model):
        original=driver.Model.forward;namespace=dict(original.__globals__)
        def features(*args,**kwargs):kwargs['eos']=None;return self.token_features(*args,**kwargs)
        namespace['token_features']=features
        fn=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__
        return fn.__get__(model,type(model))
    def summary(self):
        rows=[]
        for d in range(self.depth):
            selected=[r for r in self.records if r['stage']==d]
            def mean(field):return sum(field(r) for r in selected)/len(selected)
            rows.append(dict(stage=d,feature_change_rms=mean(lambda r:float((r['output'].detach()-r['input'].detach()).square().mean().sqrt())),delay_mean=mean(lambda r:float((r['delivery']-r['arrival']).detach().mean())),input_cotangent_rms=mean(lambda r:float(r['input'].grad.square().mean().sqrt()) if r['input'].grad is not None else 0.),output_cotangent_rms=mean(lambda r:float(r['output'].grad.square().mean().sqrt()) if r['output'].grad is not None else 0.)))
        return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--run',default='aws_integrated_receiver_d8_8m_20261014_s6_v1');a=ap.parse_args();torch.set_num_threads(1)
    base=ROOT/f'experiments/results/token_language/{a.run}';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed'
    for name,digest in r['source_sha256'].items():assert driver.sha(ROOT/name)==digest,name
    args=SimpleNamespace(**r['args']);args.compiled=False
    train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file);counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
    torch.manual_seed(args.seed);model=driver.Model(args,torch.argsort(counts,descending=True,stable=True),counts);ck=torch.load(base.with_suffix('.pt'),weights_only=False);model.load_state_dict(ck['model']);del ck;model.eval()
    # Existing reused development interval, no public or fresh selection data.
    ids=driver.interval_tensor(dev,25165824,520,8);inputs=ids[:,:64];labels=ids[:,1:65];assert labels.shape==inputs.shape
    start=time.monotonic();seed=args.seed+100000
    def evaluate(audit,grad=False):
        model.zero_grad(set_to_none=True);g=torch.Generator().manual_seed(seed)
        with torch.set_grad_enabled(grad):
            features,state,_=audit.forward(model)(inputs,None,g,args,deterministic=args.deterministic_eval)
            loss=model.readout.nll(features,labels).mean()
            if grad:loss.backward()
        return float(loss.detach()),features.detach(),state,g.get_state()
    observe=StageAudit(args.depth);nll,features,state,tape=evaluate(observe,True)
    with torch.no_grad():
        factual,reference_state,_=reference_forward(model,driver)(inputs,None,torch.Generator().manual_seed(seed),args,deterministic=args.deterministic_eval)
    assert torch.equal(features,factual),'Instrumented factual features changed'
    assert all(torch.equal(x,y) for x,y in zip(state['mem'],reference_state['mem']))
    profile=observe.summary();del observe
    variants=[]
    for mode in ('gain_zero','bypass'):
        for d in range(args.depth):
            audit=StageAudit(args.depth,mode,d);changed,f,s,other_tape=evaluate(audit)
            assert torch.equal(tape,other_tape),'Intervention changed paired random draws'
            variants.append(dict(mode=mode,stage=d,nll=changed,delta_nll=changed-nll,final_feature_difference_rms=float((f-features).square().mean().sqrt()),clock_context_difference=float((s['ctx_arr']-state['ctx_arr']).abs().mean())))
    names=['experiments/credit/native_depth_causal_audit.py','sleeping_machines/ordered_token_episodes.py','sleeping_machines/sparse_counterfactual_episodes.py','sleeping_machines/sparse_counterfactual_layer.py','experiments/public_token_scoring.py']
    sources=dict(r['source_sha256']);sources.update({n:driver.sha(ROOT/n) for n in names})
    output=dict(status='completed',battle='R1',parent=a.run,baseline_nll=nll,profile=profile,interventions=variants,factual_parity=True,paired_random_draws=True,targets=labels.numel(),source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen acquiredD8,512reusedDEVtargets/64history,one paired race tape; factual cotangents truncated at declared16token boundaries; stage interventions change recurrent dynamics at fixeddepth. No optimizer updates, fitting, public scoring or causal proof of training-depth deficit. Directional mechanism evidence guides a contracted repair.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output),flush=True)
if __name__=='__main__':main()
