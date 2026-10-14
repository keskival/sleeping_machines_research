"""Batch16 with exact loss-weighted microbatch4 accumulation for all families.

One global clip/AdamW operation per logical batch. Reduces attention peak RSS
without changing the information/history boundary or granting extra updates.
"""
import torch

def update(model,opt,t,m):
    model.train();opt.zero_grad(set_to_none=True);loss_value=0.
    for start in range(0,len(t),4):
        stop=min(start+4,len(t));weight=(stop-start)/len(t)
        loss=-model.terms(t[start:stop],m[start:stop]).mean()*weight
        assert torch.isfinite(loss),'nonfinite loss'
        loss.backward();loss_value+=float(loss.detach())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'nonfinite gradients'
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);opt.step()
    return loss_value
