"""Deep native event computation with receiver-local forward eligibility.

Every layer owns horizontal persistent receivers. No top-to-bottom recurrent
loop and no temporal autograd tape. Higher layers consume detached messages;
each layer must receive a local loss/readout cotangent. Scoring all pool keys
and propagating their local eligibility is paid; capacity is not free.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .sparse_race_language import TemporalUnit
from .sparse_training import _transport
from .native_receiver_eligibility import ReceiverEligibility


class LocalDeepRace(nn.Module):
    def __init__(self,width=64,depth=8,heads=4,pool=16):
        super().__init__()
        if min(width,depth,heads,pool)<1 or width%heads or (width//heads)%2:
            raise ValueError('positive dimensions and even head payload required')
        self.width,self.depth,self.heads,self.pool=width,depth,heads,pool
        self.payload=width//heads;p=self.payload
        self.mix=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(depth))
        for m in self.mix:nn.init.eye_(m.weight)
        self.query=nn.ModuleList(nn.ModuleList(nn.Linear(width,p,bias=False) for _ in range(heads)) for _ in range(depth))
        self.units=nn.ModuleList(nn.ModuleList(nn.ModuleList(TemporalUnit(p,.5/math.sqrt(depth)) for _ in range(pool)) for _ in range(heads)) for _ in range(depth))
        tau=torch.logspace(0,2,p//2)
        self.transport_rate=nn.Parameter(torch.expm1(1/tau).log().repeat(depth,heads,1))
        self.transport_frequency=nn.Parameter(torch.linspace(-math.pi/2,math.pi/2,p//2).repeat(depth,heads,1))

    def new_state(self):
        return dict(receivers={},events=0,candidate_scores=0,selected_updates=0,
                    alternative_values=0,credit_projections=0)

    @staticmethod
    def value(unit,incoming,memory):
        y=F.layer_norm(unit.output(memory)+incoming,(unit.payload,))
        return incoming+unit.gain*y*torch.sigmoid(unit.gate(F.gelu(y)))

    def consume(self,message,state,generator,alternative_generator,*,learn=True):
        if message.shape!=(self.width,) or not bool(torch.isfinite(message).all()):
            raise ValueError('finite fixed-width native event message required')
        if message.device.type!='cpu':raise ValueError('initial event kernel is CPU-only')
        x=message;arrival=torch.tensor(float(state['events']),dtype=torch.float64)
        outputs=[];packets=[];winners=[]
        for d in range(self.depth):
            # Explicit local interface: numerical information persists, gradient
            # graphs do not cross a message boundary into a different layer.
            if d:x=x.detach();arrival=arrival.detach()
            stream=x;mixed=self.mix[d](x);features=F.layer_norm(mixed,(self.width,))
            values=[];times=[];level_winners=[]
            for h in range(self.heads):
                units=self.units[d][h];incoming=mixed[h*self.payload:(h+1)*self.payload]
                q=self.query[d][h](features);old=[];memories=[];scores=[]
                for u,unit in enumerate(units):
                    trace=state['receivers'].get((d,h,u))
                    if trace is None:trace=ReceiverEligibility.empty(unit)
                    old.append(trace);m=trace.memory().requires_grad_(learn);memories.append(m)
                    scores.append(q@(unit.key+unit.key_read(m))/math.sqrt(self.payload)+unit.clock_bias)
                    if learn:packets.append((unit,trace,m))
                scores=torch.stack(scores).clamp(-12,12);rates=scores.double().exp()
                noise=torch.empty(self.pool,dtype=torch.float64).exponential_(generator=generator)
                first,winner=(noise/rates.detach()).min(0);u=int(winner);unit=units[u]
                proposed=old[u].propose(unit,incoming,arrival.detach())
                memory=proposed.memory().requires_grad_(learn)
                value=self.value(unit,incoming,memory)
                if learn:
                    packets.append((unit,proposed,memory))
                    if self.pool>1:
                        pi=rates/rates.sum();alt_noise=torch.empty(self.pool,dtype=torch.float64).exponential_(generator=alternative_generator)
                        alt_times=alt_noise/pi.detach();alt_times[u]=float('inf');j=int(alt_times.argmin())
                        with torch.no_grad():
                            stamp=None if old[j].timestamp is None else arrival.new_tensor(old[j].timestamp)
                            alternative,_=units[j].propose(incoming.detach(),old[j].memory(),stamp,arrival.detach())
                        probability=pi[j]/(1-pi[u]);coefficient=((pi[j]-pi[j].detach())/probability.detach()).to(value.dtype)
                        value=value+coefficient*(alternative-value).detach()
                        state['alternative_values']+=1
                # Native factorized-clock local credit; no graph of past clocks.
                clock=first-first*(rates.sum().log()-rates.sum().log().detach())
                delay=.001+.010*clock/(1+clock)
                values.append(value);times.append(arrival+delay)
                state['receivers'][(d,h,u)]=proposed
                state['candidate_scores']+=self.pool;state['selected_updates']+=1;level_winners.append(u)
            ready=torch.stack(times).max()
            aligned=torch.cat([_transport(v[None],(ready-t).reshape(1),self.transport_rate[d,h],self.transport_frequency[d,h])[0] for h,(v,t) in enumerate(zip(values,times))])
            x=F.layer_norm(stream+aligned/math.sqrt(self.depth),(self.width,));arrival=ready
            outputs.append(x);winners.append(level_winners)
        state['events']+=1
        return torch.stack(outputs),state,packets,dict(winners=winners,ready=float(arrival.detach()))

    @staticmethod
    @torch.no_grad()
    def deliver_credit(packets,state):
        """Apply local state cotangents after local losses mature; no replay."""
        grouped={}
        for unit,trace,leaf in packets:
            if leaf.grad is None or trace.timestamp is None:continue
            grouped.setdefault(unit,[]).append((trace,leaf.grad))
        for unit,records in grouped.items():
            for name,credit in ReceiverEligibility.sum_credit(records).items():
                parameter=dict(unit.named_parameters())[name]
                if parameter.grad is None:parameter.grad=credit.clone()
                else:parameter.grad.add_(credit)
            state['credit_projections']+=len(records)

    def storage(self,state):
        return dict(occupied_receivers=len(state['receivers']),
                    receiver_state_and_eligibility_bytes=sum(s.storage()['tensor_bytes'] for s in state['receivers'].values()),
                    temporal_graph_entries=0,
                    scope='Persistent numerical receiver state plus own-write traces;current event graphs,credit packets,parameters and metadata separate.')
