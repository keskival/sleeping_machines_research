"""Packed execution of LocalDeepRace; same native model and local credit law."""
import math
import torch
from torch import nn
from torch.nn import functional as F

FIELDS={'key':'key','key_read':'key_read.weight','clock_bias':'clock_bias',
        'input':'input.weight','output':'output.weight','gate_w':'gate.weight','gate_b':'gate.bias',
        'control_w':'control.weight','control_b':'control.bias','raw_rate':'raw_rate','frequency':'frequency'}
TRACES=('input_real','input_imag','control_weight','control_bias','raw_rate','frequency')
def field(unit,path):
    value=unit
    for part in path.split('.'):value=getattr(value,part)
    return value


class PackedLocalDeepRace(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.width,self.depth,self.heads,self.pool,self.payload=(getattr(source,k) for k in ['width','depth','heads','pool','payload'])
        self.gain=.5/math.sqrt(self.depth)
        self.banks=nn.ParameterDict({name:nn.Parameter(torch.stack([torch.stack([torch.stack([field(u,path).detach() for u in units]) for units in layer]) for layer in source.units])) for name,path in FIELDS.items()})
        self.mix=nn.Parameter(torch.stack([m.weight.detach() for m in source.mix]))
        self.query=nn.Parameter(torch.stack([torch.stack([q.weight.detach() for q in layer]) for layer in source.query]))
        self.transport_rate=nn.Parameter(source.transport_rate.detach().clone())
        self.transport_frequency=nn.Parameter(source.transport_frequency.detach().clone())

    def new_state(self):
        h,u,m,p=self.heads,self.pool,self.payload//2,self.payload
        dtype=torch.complex128 if self.mix.dtype==torch.float64 else torch.complex64
        z=lambda *shape:torch.zeros(*shape,dtype=dtype,device=self.mix.device)
        layers=[]
        for _ in range(self.depth):
            layers.append(dict(value=z(h,u,m),input_real=z(h,u,m,p),input_imag=z(h,u,m,p),
                control_weight=z(h,u,m,2,p),control_bias=z(h,u,m,2),raw_rate=z(h,u,m),frequency=z(h,u,m),
                time=torch.zeros(h,u,dtype=torch.float64),seen=torch.zeros(h,u,dtype=torch.bool)))
        return dict(layers=layers,events=0,candidate_scores=0,selected_updates=0,alternative_values=0,credit_projections=0)

    @torch.no_grad()
    def proposal(self,old,params,incoming,elapsed,*,traces):
        x=incoming.detach();normal=F.layer_norm(x,(self.payload,))
        controls=torch.einsum('hcp,hp->hc',params['control_w'],normal)+params['control_b']
        forget=F.softplus(controls[:,0])/math.log(2);sig=controls[:,1].sigmoid();write=2*sig
        rate=F.softplus(params['raw_rate'])+1e-6
        magnitude=torch.exp(-elapsed.to(x.dtype)[:,None]*rate*forget[:,None]);angle=elapsed[:,None]*params['frequency'].double()
        rotation=torch.complex(magnitude*angle.cos().to(x.dtype),magnitude*angle.sin().to(x.dtype))
        transported=rotation*old['value'];b=torch.einsum('hpq,hq->hp',params['input'],x).reshape(self.heads,-1,2)
        b=torch.complex(b[...,0],b[...,1]);result=dict(value=transported+write[:,None]*b)
        if not traces:return result
        cf=transported*(-elapsed.to(x.dtype)[:,None]*rate*controls[:,0].sigmoid()[:,None]/math.log(2))
        cw=b*(2*sig*(1-sig))[:,None];dc=torch.stack((cf,cw),-1)
        result.update(input_real=rotation[...,None]*old['input_real']+write[:,None,None]*x[:,None],
            input_imag=rotation[...,None]*old['input_imag']+1j*write[:,None,None]*x[:,None],
            control_weight=rotation[...,None,None]*old['control_weight']+dc[...,None]*normal[:,None,None],
            control_bias=rotation[...,None]*old['control_bias']+dc,
            raw_rate=rotation*old['raw_rate']-transported*elapsed.to(x.dtype)[:,None]*forget[:,None]*params['raw_rate'].sigmoid(),
            frequency=rotation*old['frequency']+1j*elapsed.to(x.dtype)[:,None]*transported)
        return result

    def unit_value(self,params,x,m):
        y=F.layer_norm(torch.einsum('hpq,hq->hp',params['output'],m)+x,(self.payload,))
        gate=torch.einsum('hpq,hq->hp',params['gate_w'],F.gelu(y))+params['gate_b']
        return x+self.gain*y*gate.sigmoid()

    def consume(self,message,state,generator,alternative_generator,*,learn=True,noise=None,alternative_noise=None,cause_messages=None):
        h,u,p=self.heads,self.pool,self.payload;hi=torch.arange(h);x=message;arrival=torch.tensor(float(state['events']),dtype=torch.float64)
        outputs=[];packets=[];winners=[]
        if cause_messages is not None and cause_messages.shape!=(self.depth,self.width):
            raise ValueError('one cause-message inbox per native depth required')
        for d in range(self.depth):
            if d:x=x.detach();arrival=arrival.detach()
            if cause_messages is not None:x=x+cause_messages[d].detach()
            stream=x;mixed=F.linear(x,self.mix[d]);normal=F.layer_norm(mixed,(self.width,));incoming=mixed.reshape(h,p)
            input_arrival=arrival.detach()
            old=state['layers'][d];b={k:v[d] for k,v in self.banks.items()}
            memory=torch.view_as_real(old['value']).flatten(-2).detach().requires_grad_(learn)
            keys=b['key']+torch.einsum('hupq,huq->hup',b['key_read'],memory)
            query=torch.einsum('hpw,w->hp',self.query[d],normal)
            scores=((query[:,None]*keys).sum(-1)/math.sqrt(p)+b['clock_bias']).clamp(-12,12);rates=scores.double().exp()
            draws=torch.empty(h,u,dtype=torch.float64).exponential_(generator=generator) if noise is None else noise[d]
            first,winner=(draws/rates.detach()).min(-1);selected={k:v[hi,winner] for k,v in b.items()}
            previous={k:v[hi,winner] for k,v in old.items()}
            elapsed=torch.where(previous['seen'],arrival.detach()-previous['time'],torch.zeros_like(previous['time']))
            proposed=self.proposal(previous,selected,incoming,elapsed,traces=True)
            new_memory=torch.view_as_real(proposed['value']).flatten(-2).detach().requires_grad_(learn)
            value=self.unit_value(selected,incoming,new_memory)
            if learn:
                packets.append((d,old,memory,proposed,new_memory,winner))
                if u>1:
                    pi=rates/rates.sum(-1,keepdim=True)
                    draws_alt=torch.empty(h,u,dtype=torch.float64).exponential_(generator=alternative_generator) if alternative_noise is None else alternative_noise[d]
                    alt_times=(draws_alt/pi.detach()).scatter(1,winner[:,None],float('inf'));alt=alt_times.argmin(-1)
                    with torch.no_grad():
                        ap={k:v[hi,alt] for k,v in b.items()};ast={k:v[hi,alt] for k,v in old.items()}
                        adt=torch.where(ast['seen'],arrival.detach()-ast['time'],torch.zeros_like(ast['time']))
                        am=self.proposal(ast,ap,incoming,adt,traces=False)['value']
                        alternative=self.unit_value(ap,incoming,torch.view_as_real(am).flatten(-2))
                    pa=pi[hi,alt];pw=pi[hi,winner];coefficient=((pa-pa.detach())/(pa/(1-pw)).detach()).to(value.dtype)
                    value=value+coefficient[:,None]*(alternative-value).detach();state['alternative_values']+=h
            log_rate=rates.sum(-1).log();clock=first-first*(log_rate-log_rate.detach())
            times=arrival+.001+.010*clock/(1+clock);ready=times.max();age=ready-times
            rate=F.softplus(self.transport_rate[d])+1e-6;mag=torch.exp(-age.to(x.dtype)[:,None]*rate)
            angle=age[:,None]*self.transport_frequency[d].double();pairs=value.reshape(h,-1,2)
            real,imag=pairs[...,0]*mag,pairs[...,1]*mag;c,s=angle.cos().to(x.dtype),angle.sin().to(x.dtype)
            aligned=torch.stack((c*real-s*imag,s*real+c*imag),-1).flatten()
            x=F.layer_norm(stream+aligned/math.sqrt(self.depth),(self.width,));arrival=ready
            mask=F.one_hot(winner,u).bool();updated={}
            for name in ['value',*TRACES]:
                v=old[name];shape=mask.shape+(1,)*(v.ndim-2)
                updated[name]=torch.where(mask.reshape(shape),proposed[name][:,None],v)
            # Receiver writes retain stage input time, not emission time.
            updated['time']=torch.where(mask,input_arrival.expand(h,u),old['time']);updated['seen']=old['seen']|mask
            state['layers'][d]=updated;state['candidate_scores']+=h*u;state['selected_updates']+=h
            outputs.append(x);winners.append(winner.detach().tolist())
        state['events']+=1
        return torch.stack(outputs),state,packets,dict(winners=winners,ready=float(arrival.detach()))

    @torch.no_grad()
    def deliver_credit(self,packets,state):
        def project(trace,g):
            pair=g.reshape(*g.shape[:-1],-1,2);cot=torch.complex(pair[...,0],pair[...,1]).conj()
            def v(name):
                t=trace[name];return (cot.reshape(cot.shape+(1,)*(t.ndim-cot.ndim))*t).real
            return {'input':torch.stack((v('input_real'),v('input_imag')),-2).flatten(-3,-2),
                'control_w':v('control_weight').sum(-3),'control_b':v('control_bias').sum(-2),
                'raw_rate':v('raw_rate'),'frequency':v('frequency')}
        hi=torch.arange(self.heads)
        for d,old,memory,new,new_memory,winner in packets:
            for trace,g,selected in [(old,memory.grad,False),(new,new_memory.grad,True)]:
                if g is None:continue
                for name,credit in project(trace,g).items():
                    bank=self.banks[name]
                    if bank.grad is None:bank.grad=torch.zeros_like(bank)
                    if selected:bank.grad[d,hi,winner]+=credit
                    else:bank.grad[d]+=credit
                state['credit_projections']+=self.heads if selected else int(old['seen'].sum())

    def storage(self,state):
        return dict(occupied_receivers=sum(int(s['seen'].sum()) for s in state['layers']),
            receiver_state_and_eligibility_bytes=sum(v.numel()*v.element_size() for s in state['layers'] for k,v in s.items() if k not in ['time','seen']),
            temporal_graph_entries=0,scope='Dense reserved receiver/trace tensors;old trace snapshots,credit packets,parameters,metadata and timestamps additional.')
