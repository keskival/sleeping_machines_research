"""Analytic four-direction shared-path sensitivities; fixed-parameter CPU contract backend."""
import math
import numpy as np


def array(t):
    return t.detach().numpy()


def norm_tangent(module, x, tangent):
    centered=x-x.mean();variance=np.mean(centered**2)
    return array(module.weight)[:,None]/math.sqrt(variance+module.eps)*(tangent-tangent.mean(0)-centered[:,None]*np.mean(centered[:,None]*tangent,0)/(variance+module.eps))


class ProjectedClockState:
    def __init__(self,model,state):
        assert not model.training and not hasattr(model,'key_w')
        self.model=model;self.version=state['parameter_version'];self.rank=2*len(model.layers)
        self.sensitivity=[np.zeros((z.numel(),self.rank)) for z in state['layers']]+[np.zeros((state['slots'].numel(),self.rank))]
        self.parameter_versions=tuple(p._version for p in model.parameters())
    def advance(self,time,mark,state,consumer):
        assert self.parameter_versions==tuple(p._version for p in self.model.parameters()),'Reset after parameter update'
        self.previous=[x.copy() for x in self.sensitivity];capture={};hooks=[]
        def remember(key):
            def hook(module,args,out):capture[key]=(array(args[0])[0].copy(),array(out)[0].copy())
            return hook
        for j,l in enumerate(self.model.layers):
            for key,module in [('write',l.write),('gate',l.gate),('norm1',l.norm1),('mlp0',l.mlp[0]),('norm2',l.norm2)]:hooks.append(module.register_forward_hook(remember((j,key))))
        hooks.append(self.model.marks.value.register_forward_hook(remember('value')))
        try:forecast,end=consumer(self.model,time,mark,state)
        finally:
            for h in hooks:h.remove()
        dt=float((time-state['last_time'])[0])/self.model.scale;dh=np.zeros((self.model.embed.embedding_dim,self.rank));new=[]
        for j,l in enumerate(self.model.layers):
            n=l.n;old=array(state['layers'][j])[0];r,i=old[:n],old[n:];sr,si=self.previous[j][:n],self.previous[j][n:]
            rate=np.exp(array(l.log_rate));angle=array(l.freq)*dt;decay=np.exp(-rate*dt);cr=decay*np.cos(angle);ci=decay*np.sin(angle)
            rotated_r=cr*r-ci*i;rotated_i=ci*r+cr*i
            dr=cr[:,None]*sr-ci[:,None]*si;di=ci[:,None]*sr+cr[:,None]*si
            dr[:,2*j]+=-dt*rate*rotated_r/math.sqrt(n);di[:,2*j]+=-dt*rate*rotated_i/math.sqrt(n)
            dr[:,2*j+1]+=-dt*rotated_i/math.sqrt(n);di[:,2*j+1]+=dt*rotated_r/math.sqrt(n)
            w=capture[j,'write'][1];g=1/(1+np.exp(-capture[j,'gate'][1]));dw=array(l.write.weight)@dh;dg=g[:,None]*(1-g[:,None])*(array(l.gate.weight)@dh)
            dz=np.concatenate((dr,di),0)+np.tile(g,2)[:,None]*dw+w[:,None]*np.concatenate((dg,dg),0);new.append(dz)
            dh=norm_tangent(l.norm1,capture[j,'norm1'][0],dh+array(l.read.weight)@dz)
            raw=capture[j,'mlp0'][1]
            derivative=.5*(1+np.array([math.erf(float(v)/math.sqrt(2)) for v in raw]))+raw*np.exp(-raw**2/2)/math.sqrt(2*math.pi)
            assert l.mlp[1].approximate=='none'
            dh=norm_tangent(l.norm2,capture[j,'norm2'][0],dh+array(l.mlp[2].weight)@(derivative[:,None]*(array(l.mlp[0].weight)@dh)))
        slot_old=self.previous[-1].reshape(self.model.K,self.model.marks.dv,self.rank)
        rho=np.exp(-np.exp(array(self.model.marks.log_rate))*dt);slots=slot_old*rho[None,:,None]
        raw=capture['value'][1];slots[int(mark[0])]+=1/(1+np.exp(-raw[:,None]))*(array(self.model.marks.value.weight)@dh)
        new.append(slots.reshape(-1,self.rank));self.sensitivity=new
        return forecast,end
    def historical_projection(self,old_cotangents):
        return sum((s.T@array(c).reshape(-1) for s,c in zip(self.previous,old_cotangents) if c is not None),np.zeros(self.rank))
    def project_parameters(self,grads):
        result=[]
        for j,l in enumerate(self.model.layers):
            for key in ['log_rate','freq']:
                result.append(float(grads[f'layers.{j}.{key}'].sum())/math.sqrt(l.n))
        return np.array(result)
