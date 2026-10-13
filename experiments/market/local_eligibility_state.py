"""Own-leg eligibility observer for the unchanged saved temporal market actor.
Serial CPU prototype; releases each event's graph. Missing cross-leg/history
paths remain explicit. Traces require fixed parameters within a declared window.
"""
import numpy as np
import torch
from experiments.credit.local_clock_eligibility import LocalClockEligibility
class MarketLegEligibility:
    def __init__(self,model,state,consumer,version=0):
        self.model=model;self.consumer=consumer;self.version=version;self.legs=[];self.history=[]
        for layer,z in zip(model.layers,state['layers']):
            n=layer.n;w=layer.write.weight.detach().numpy();b=layer.write.bias.detach().numpy()
            gate=np.column_stack((layer.gate.weight.detach().numpy(),layer.gate.bias.detach().numpy()))
            leg=LocalClockEligibility(np.column_stack((w[:n],b[:n])),np.column_stack((w[n:],b[n:])),gate,layer.log_rate.detach().numpy(),layer.freq.detach().numpy())
            r,i=z.detach().numpy()[0].reshape(2,n);leg.state=np.stack((r,i),-1);self.legs.append(leg)
        self.value=np.column_stack((model.marks.value.weight.detach().numpy(),model.marks.value.bias.detach().numpy()))
        self.slot_rates=np.exp(model.marks.log_rate.detach().numpy());self.slot_trace=np.zeros((model.K,model.marks.dv,self.value.shape[1]+1));self.slot_history=np.zeros_like(self.slot_trace)
        self.physical_error=0.
    def consume(self,time,mark,state,version=0):
        assert version==self.version,'Reset/version traces after parameter changes'
        capture={};hooks=[]
        def remember(key):
            def hook(module,args,out):capture[key]=args[0].detach().numpy()[0].copy()
            return hook
        for index,layer in enumerate(self.model.layers):hooks.append(layer.write.register_forward_hook(remember(index)))
        hooks.append(self.model.marks.value.register_forward_hook(remember('value')))
        try:forecast,end=self.consumer(self.model,time,mark,state,parameter_version=version)
        finally:
            for hook in hooks:hook.remove()
        dt=float((time-state['last_time'])[0])/self.model.scale;self.history=[]
        for index,(layer,leg) in enumerate(zip(self.model.layers,self.legs)):
            n=layer.n;r,i=state['layers'][index].detach().numpy()[0].reshape(2,n);leg.state=np.stack((r,i),-1)
            decay=np.exp(-np.exp(leg.log_rate)*dt);angle=leg.frequency*dt;c=decay*np.cos(angle);s=decay*np.sin(angle)
            rotation=np.stack((np.stack((c,-s),-1),np.stack((s,c),-1)),1)
            self.history.append(np.einsum('nij,njp->nip',rotation,leg.trace))
            predicted=leg.consume(np.append(capture[index],1.),dt);actual=end['layers'][index].detach().numpy()[0].reshape(2,n).T
            self.physical_error=max(self.physical_error,float(np.max(np.abs(predicted-actual))))
        rho=np.exp(-self.slot_rates*dt);self.slot_history=rho[None,:,None]*self.slot_trace;self.slot_trace=self.slot_history.copy()
        x=np.append(capture['value'],1.);raw=self.value@x;gate=1/(1+np.exp(-raw));source=int(mark[0])
        self.slot_trace[source,:,:-1]+=gate[:,None]*x
        old=state['slots'].detach().numpy()[0];self.slot_trace[:,:,-1]+=-dt*self.slot_rates[None,:]*rho[None,:]*old
        predicted=old*rho[None,:];predicted[source]+=np.logaddexp(0,raw)
        self.physical_error=max(self.physical_error,float(np.max(np.abs(predicted-end['slots'].detach().numpy()[0]))))
        return forecast,end
    def historical_gradient(self,cotangents):
        out={}
        for index,(leg,history,cot) in enumerate(zip(self.legs,self.history,cotangents[:-1])):
            n=leg.units;d=leg.input_width-1;c=cot.detach().numpy()[0].reshape(2,n).T
            g=np.einsum('ni,nip->np',c,history);wr=g[:,:d+1];wi=g[:,d+1:2*(d+1)];gate=g[:,2*(d+1):3*(d+1)];prefix=f'layers.{index}.'
            out[prefix+'write.weight']=np.concatenate((wr[:,:-1],wi[:,:-1]),0);out[prefix+'write.bias']=np.concatenate((wr[:,-1],wi[:,-1]))
            out[prefix+'gate.weight']=gate[:,:-1];out[prefix+'gate.bias']=gate[:,-1];out[prefix+'log_rate']=g[:,-2];out[prefix+'freq']=g[:,-1]
        g=np.einsum('kv,kvp->vp',cotangents[-1].detach().numpy()[0],self.slot_history)
        out['marks.value.weight']=g[:,:-2];out['marks.value.bias']=g[:,-2];out['marks.log_rate']=g[:,-1]
        return out
    def storage(self):
        return dict(eligibility_floats=sum(x.trace.size for x in self.legs)+self.slot_trace.size,historical_projection_floats=sum(x.size for x in self.history)+self.slot_history.size,scope='Own eligibility plus separate history projection workspace; copied parameters, gradients, physical state and current frame are additional.')
