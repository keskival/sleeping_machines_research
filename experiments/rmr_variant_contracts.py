import sys, math, torch, numpy as np; sys.path[:0]=[__import__('os').path.dirname(__file__), __import__('os').path.dirname(__file__)+'/tpp']
import r1_rmr as R
from r1_cerl import word_classes
torch.manual_seed(0); V=61; cls,words=word_classes(np.random.default_rng(0).integers(0,50,V).astype(float),6,s_max=16)
cerl=dict(d=32,modes=8,layers=1,dk=16,units=32,topk=3,explore=0)
for name,kw in (('dense_stack',dict(top_m=0,read_topk=0)),('sparse',dict(top_m=3,read_topk=4)),('old_decay',dict(top_m=3,read_topk=0,old_decay=0.05))):
    m=R.RMR(V,cls,words,cerl,levels=3,dk=16,chunk=8,**kw); m.eval()
    with torch.no_grad():
        x=torch.randint(0,8,(1,40)); y=torch.cat([x[:,1:],torch.randint(0,8,(1,1))],1)
        z=float((m.lane_nll(x.clone(),y)-m.base.lane_nll(x,y)).abs().max())
        for p in m.parameters(): p.add_(torch.randn_like(p)*0.1)
        full=m.lane_nll(x.clone(),y); tot=sum(math.exp(-m.lane_nll(x.clone(),torch.cat([y[:,:25],torch.tensor([[v]]),y[:,26:]],1))[0,25].item()) for v in range(V))
        x2,y2=x.clone(),y.clone(); x2[0,30]=(x2[0,30]+1)%V; y2[0,29]=x2[0,30]; c=float((m.lane_nll(x2,y2)[0,:29]-full[0,:29]).abs().max())
    print(name, 'zero_eq', z, 'norm', round(tot,7), 'causal', c)
