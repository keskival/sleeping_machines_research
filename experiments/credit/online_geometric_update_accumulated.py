"""Causal learned geometric update; retained credit graph receives next loss."""
import math
import torch

class UpdateFunction(torch.nn.Module):
    def __init__(self,context_size,width=64):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(context_size,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,3))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,context):
        raw=self.net(context);eta=.01*torch.exp(math.log(10)*raw[0].tanh());damping=torch.exp(math.log(10)*raw[1].tanh());beta=.99*torch.sigmoid(raw[2]+math.log(10))
        return eta,damping,beta

def geometric_step(z,moment,beta_product,gradient,curvature,eta,damping,beta):
    d=len(z);scale=curvature.diag().mean().detach().clamp_min(1e-8)
    updated_m=beta*moment+(1-beta)*gradient;updated_product=beta*beta_product
    direction=updated_m/(1-updated_product).clamp_min(1e-8)
    matrix=curvature+damping*scale*torch.eye(d,dtype=z.dtype,device=z.device)
    delta=eta*torch.linalg.solve(matrix,direction)
    delta=delta*(delta.new_tensor(.08)/delta.norm().clamp_min(1e-12)).clamp_max(1)
    next_z=z-delta
    return next_z,updated_m,updated_product

def update_context(hidden,z,moment,gradient,diagonal,nll,position):
    norm=lambda x:x/x.square().mean().sqrt().clamp_min(1e-8)
    return torch.cat([norm(hidden),z.detach(),norm(moment),norm(gradient),torch.log1p(diagonal.clamp_min(0)/diagonal.mean().clamp_min(1e-8)),z.new_tensor([nll,math.log1p(position)])]).detach()


class OnlineUpdateLearner:
    def __init__(self,function,dimension,train_credit=False,optimizer_state=None,feedback_batch=16):
        self.function=function;self.train_credit=train_credit;self.optimizer=torch.optim.Adam(function.parameters(),lr=.001)
        if optimizer_state is not None:self.optimizer.load_state_dict(optimizer_state)
        self.version=0;self.meta_updates=0;self.meta_feedbacks=0;self.accumulated=0;self.feedback_batch=feedback_batch;assert feedback_batch>0;self.reset(dimension)
    def reset(self,dimension=None):
        if dimension is None:dimension=len(self.z)
        self.z=next(self.function.parameters()).new_zeros(dimension);self.moment=torch.zeros_like(self.z);self.product=self.z.new_tensor(1.);self.pending=None;self.pending_version=None
    def commit_feedback(self):
        if not self.accumulated:return
        for parameter in self.function.parameters():
            if parameter.grad is not None:parameter.grad.div_(self.accumulated)
        self.optimizer.step();self.optimizer.zero_grad();self.version+=1;self.meta_updates+=1;self.accumulated=0
    def finish(self):
        self.pending=None;self.pending_version=None;self.commit_feedback()
    def feedback(self,gradient,curvature,hidden,nll,position):
        if self.train_credit and self.pending is not None:
            if self.pending_version!=self.version:raise ValueError('Credit version changed before feedback')
            assert torch.equal(self.pending.detach(),self.z)
            if self.accumulated==0:self.optimizer.zero_grad()
            self.pending.backward(gradient.detach());self.accumulated+=1;self.meta_feedbacks+=1;self.pending=None
            if self.accumulated==self.feedback_batch:self.commit_feedback()
        context=update_context(hidden,self.z,self.moment,gradient,curvature.diag(),nll,position)
        with torch.set_grad_enabled(self.train_credit):
            eta,damping,beta=self.function(context);next_z,moment,product=geometric_step(self.z,self.moment,self.product,gradient.detach(),curvature.detach(),eta,damping,beta)
        self.z=next_z.detach();self.moment=moment.detach();self.product=product.detach()
        self.pending=next_z if self.train_credit else None;self.pending_version=self.version if self.train_credit else None
        assert bool(torch.isfinite(self.z).all())
        return dict(eta=float(eta.detach()),damping=float(damping.detach()),beta=float(beta.detach()),z_norm=float(self.z.norm()))
