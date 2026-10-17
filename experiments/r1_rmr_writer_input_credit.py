"""Explicit full first-order writer-input derivative, separate from old runs.

Sparse write algebra is unchanged; extra autograd storage must pass a resource
 gate before long-history use. No claim of constant learning memory.
"""
from experiments.r1_rmr import RMR

class WriterInputCreditRMR(RMR):
    def slot_values(self,base_prime,wr,idx,rows):
        batch,count,slots=idx.shape;length=base_prime.shape[1]
        reads=base_prime.new_zeros(batch,count*slots,length).scatter_(2,idx.reshape(batch,count*slots,1),1.)
        correction=self.corr_values(reads,wr,rows.repeat_interleave(slots)).to(base_prime.dtype)
        value=base_prime.gather(1,idx.reshape(batch,-1,1).expand(-1,-1,base_prime.shape[-1]))+correction
        return value.reshape(batch,count,slots,-1)


class CheckpointedWriterInputCreditRMR(WriterInputCreditRMR):
    """Exact local recomputation instead of retaining the large gather graph."""
    def slot_values(self,base_prime,wr,idx,rows):
        import torch
        from torch.utils.checkpoint import checkpoint
        batch,count,slots=idx.shape;length=base_prime.shape[1]
        reads=base_prime.new_zeros(batch,count*slots,length).scatter_(2,idx.reshape(batch,count*slots,1),1.)
        arguments=[];layout=[]
        for write in wr:
            fields=[]
            for value in write:
                if torch.is_tensor(value):fields.append((True,len(arguments)));arguments.append(value)
                else:fields.append((False,value))
            layout.append(fields)
        def correction(*values):
            writes=[tuple(values[item] if tensor else item for tensor,item in fields) for fields in layout]
            return self.corr_values(reads,writes,rows.repeat_interleave(slots)).to(base_prime.dtype)
        if torch.is_grad_enabled() and any(t.requires_grad for t in arguments):
            old=checkpoint(correction,*arguments,use_reentrant=False,preserve_rng_state=False)
        else:old=correction(*arguments)
        value=base_prime.gather(1,idx.reshape(batch,-1,1).expand(-1,-1,base_prime.shape[-1]))+old
        return value.reshape(batch,count,slots,-1)
