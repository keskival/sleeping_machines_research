"""Same event credit function; batch shared layer decoders and typed contractions."""
import torch
from event_parameter_credit import EventCredit as ReferenceCredit,unit
from event_optimizer_credit import EventOptimizerCredit as ReferenceOptimizer

class EventCredit(ReferenceCredit):
    def forward(self,p,forced=None,pool=None):
        states=p['states'];c=unit(p['cotangent']);elapsed=torch.log1p(p['dt'])[:,None];loss=p['loss'].expand(len(c),1)
        u=torch.tanh(self.project(torch.cat([unit(states.mean(0)),c,elapsed,loss],-1)))
        reverse_dt=torch.cat([p['dt'].new_zeros(1),p['dt'][1:].flip(0)]);global_state=self.temporal(u.flip(0)[None],reverse_dt[None])[0].flip(0)
        logits=self.query(global_state).squeeze(-1)+torch.log(p['cotangent'].square().sum(-1).clamp_min(1e-16))*.1
        available=torch.ones(len(c),dtype=torch.bool,device=c.device)
        if pool is not None:available[:]=False;available[pool]=True
        chosen=[];logprob=c.new_zeros(())
        for j in range(self.queries if forced is None else len(forced)):
            ids=available.nonzero().flatten();prob=.99*torch.softmax(logits[ids],0)+.01/len(ids);a=torch.multinomial(prob,1)[0] if forced is None else (ids==forced[j]).nonzero().flatten()[0];selected=ids[a];chosen.append(selected);logprob=logprob+prob[a].log();available=available.clone();available[selected]=False
        ids=torch.stack(chosen);Q=len(ids);depth=len(states)-1;before=torch.cat([states[:1],states[:-1]],0)[:,ids];after=states[:,ids]
        position=torch.arange(depth+1,dtype=c.dtype,device=c.device)[:,None,None].expand(-1,Q,1)/max(depth,1)
        contexts=self.local(torch.cat([global_state[ids][None].expand(depth+1,-1,-1),unit(before),unit(after),c[ids][None].expand(depth+1,-1,-1),position],-1))
        groups={}
        for name,shape in zip(self.names,self.shapes):groups.setdefault(self.key(name),[]).append((name,shape,int(name.split('.')[1])+1 if name.startswith('layers.') else 0))
        result={}
        for key,group in groups.items():
            layer_ids=[v[2] for v in group];q=self.decoders[key](contexts[layer_ids]);name,shape,_=group[0]
            if name=='embed.weight':deltas=q[0].new_zeros(shape).index_add(0,p['marks'][ids],q[0])[None]/Q
            elif len(shape)==2:
                x=torch.stack([p['inputs'][n][ids] for n,_,_ in group])@self.transport[key].T;deltas=q.transpose(-1,-2)@x/Q
            else:deltas=q.mean(1)
            for (name,shape,_),delta in zip(group,deltas):result[name]=delta.reshape(shape).flatten()
        self.last_logprob=logprob;self.last_selected=ids.detach();return torch.cat([result[n] for n in self.names])

class EventOptimizerCredit(ReferenceOptimizer):
    forward=EventCredit.forward
