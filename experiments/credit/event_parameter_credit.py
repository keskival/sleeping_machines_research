"""Event-conditioned connected credit with future-trained hard query selection."""
import torch
from types import MethodType
import race_tpp_v19 as native
from temporal_scan_cpu import layer_forward

def unit(x):return x/x.square().mean(-1,keepdim=True).clamp_min(1e-16).sqrt()

def install_event_hooks(actor):
    actor.event_inputs={};actor.event_handles=[]
    for name,module in actor.named_modules():
        if isinstance(module,torch.nn.Linear) and name.startswith(('gap','layers.')):
            def hook(module,args,name=name):
                if getattr(actor,'collect_events',False):actor.event_inputs[name+'.weight']=args[0].detach().reshape(-1,args[0].shape[-1]).clone()
            actor.event_handles.append(module.register_forward_pre_hook(hook))

def packet(actor,c,loss,t):
    return dict(states=torch.stack(actor.event_states).detach().clone(),cotangent=c.detach().reshape(-1,c.shape[-1]).clone(),inputs={n:x.clone() for n,x in actor.event_inputs.items()},dt=(torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/actor.scale).reshape(-1).detach(),marks=actor.event_marks.reshape(-1).clone(),loss=loss.detach().clone())

class EventCredit(torch.nn.Module):
    def __init__(self,width,named,queries=4):
        super().__init__();self.width=width;self.queries=queries;self.names=list(named);self.shapes=[tuple(p.shape) for p in named.values()]
        self.project=torch.nn.Linear(2*width+2,64);self.temporal=native.TemporalMemoryLayer(64,16,0.);self.temporal.forward=MethodType(layer_forward,self.temporal)
        self.local=torch.nn.Sequential(torch.nn.Linear(64+3*width+1,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh());self.query=torch.nn.Linear(64,1)
        torch.nn.init.zeros_(self.query.weight);torch.nn.init.zeros_(self.query.bias)
        self.decoders=torch.nn.ModuleDict();self.transport=torch.nn.ParameterDict()
        for n,s in zip(self.names,self.shapes):
            key=self.key(n)
            if key in self.decoders:continue
            size=width if n=='embed.weight' else s[0] if len(s)==2 else int(torch.tensor(s).prod())
            self.decoders[key]=torch.nn.Linear(64,size);torch.nn.init.zeros_(self.decoders[key].weight);torch.nn.init.zeros_(self.decoders[key].bias)
            if len(s)==2 and n!='embed.weight':self.transport[key]=torch.nn.Parameter(torch.eye(s[1]))
    @staticmethod
    def key(name):return ('_'.join(name.split('.')[2:]) if name.startswith('layers.') else name.replace('.','_'))
    def forward(self,p,forced=None,pool=None):
        states=p['states'];c=unit(p['cotangent']);elapsed=torch.log1p(p['dt'])[:,None];loss=p['loss'].expand(len(c),1)
        u=torch.tanh(self.project(torch.cat([unit(states.mean(0)),c,elapsed,loss],-1)))
        reverse_dt=torch.cat([p['dt'].new_zeros(1),p['dt'][1:].flip(0)])
        global_state=self.temporal(u.flip(0)[None],reverse_dt[None])[0].flip(0)
        logits=self.query(global_state).squeeze(-1)+torch.log(p['cotangent'].square().sum(-1).clamp_min(1e-16))*.1
        available=torch.ones(len(c),dtype=torch.bool,device=c.device)
        if pool is not None:available[:]=False;available[pool]=True
        chosen=[];logprob=c.new_zeros(())
        for j in range(self.queries if forced is None else len(forced)):
            ids=available.nonzero().flatten();prob=.99*torch.softmax(logits[ids],0)+.01/len(ids)
            a=torch.multinomial(prob,1)[0] if forced is None else (ids==forced[j]).nonzero().flatten()[0]
            selected=ids[a];chosen.append(selected);logprob=logprob+prob[a].log();available=available.clone();available[selected]=False
        ids=torch.stack(chosen);Q=len(ids);updates=[];depth=len(states)-1
        contexts={}
        for layer in range(-1,depth):
            before=states[0 if layer<0 else layer];after=states[0 if layer<0 else layer+1]
            position=c.new_full((Q,1),(layer+1)/max(depth,1))
            contexts[layer]=self.local(torch.cat([global_state[ids],unit(before[ids]),unit(after[ids]),c[ids],position],-1))
        for name,shape in zip(self.names,self.shapes):
            layer=int(name.split('.')[1]) if name.startswith('layers.') else -1;key=self.key(name);q=self.decoders[key](contexts[layer])
            if name=='embed.weight':
                delta=q.new_zeros(shape).index_add(0,p['marks'][ids],q)/Q
            elif len(shape)==2:
                x=p['inputs'][name][ids]@self.transport[key].T;delta=q.T@x/Q
            else:delta=q.mean(0).reshape(shape)
            updates.append(delta.flatten())
        self.last_logprob=logprob;self.last_selected=ids.detach();return torch.cat(updates)
