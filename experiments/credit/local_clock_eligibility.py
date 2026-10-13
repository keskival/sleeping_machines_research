"""Exact factorized own-parameter traces for independent complex temporal legs.
Inputs are held fixed with respect to each leg's own parameters. Cross-leg and
input-producer derivatives require separate credit; this is not full RTRL.
"""
import numpy as np
class LocalClockEligibility:
    def __init__(self,wr,wi,gate,log_rate,frequency):
        self.wr=np.array(wr,copy=True);self.wi=np.array(wi,copy=True);self.gate=np.array(gate,copy=True)
        self.log_rate=np.array(log_rate,copy=True);self.frequency=np.array(frequency,copy=True)
        self.units,self.input_width=self.wr.shape
        self.state=np.zeros((self.units,2));self.trace=np.zeros((self.units,2,3*self.input_width+2))
    def consume(self,x,elapsed):
        x=np.asarray(x);assert x.shape==(self.input_width,) and elapsed>=0
        decay=np.exp(-np.exp(self.log_rate)*elapsed);angle=self.frequency*elapsed;c=decay*np.cos(angle);s=decay*np.sin(angle)
        rotation=np.stack((np.stack((c,-s),-1),np.stack((s,c),-1)),1)
        transported=np.einsum('nij,nj->ni',rotation,self.state)
        self.trace=np.einsum('nij,njp->nip',rotation,self.trace)
        g=1/(1+np.exp(-(self.gate@x)));real=self.wr@x;imag=self.wi@x;d=self.input_width
        self.trace[:,0,:d]+=g[:,None]*x
        self.trace[:,1,d:2*d]+=g[:,None]*x
        self.trace[:,:,2*d:3*d]+=np.stack((real,imag),-1)[:,:,None]*(g*(1-g))[:,None,None]*x
        self.trace[:,:,-2]+=-elapsed*np.exp(self.log_rate)[:,None]*transported
        self.trace[:,:,-1]+=elapsed*np.stack((-transported[:,1],transported[:,0]),-1)
        self.state=transported+g[:,None]*np.stack((real,imag),-1)
        return self.state.copy()
    def project(self,cotangent):
        out=np.einsum('ni,nip->np',np.asarray(cotangent),self.trace);d=self.input_width
        return dict(wr=out[:,:d],wi=out[:,d:2*d],gate=out[:,2*d:3*d],log_rate=out[:,-2],frequency=out[:,-1])
