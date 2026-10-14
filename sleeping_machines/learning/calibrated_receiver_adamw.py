"""Fixed TRAIN-calibrated per-bank radial caps; no global gradient reduction."""
import math
import torch
from .local_credit_banks import LocalCreditBanks
from .receiver_bank_adamw import ReceiverBankAdamW

class CalibratedLocalCreditBanks(LocalCreditBanks):
    def __init__(self,named_parameters,bank_caps,**options):
        super().__init__(named_parameters,local_clip=None,**options)
        if set(bank_caps)!=set(self.parameters):raise ValueError('Caps must cover the exact bank schema')
        self.bank_caps={b:float(c) for b,c in bank_caps.items()}
        if any(not math.isfinite(c) or c<=0 for c in self.bank_caps.values()):raise ValueError('Finite positive bank caps required')
    def apply(self,packet):
        if packet.bank not in self.parameters:raise ValueError('Unknown bank')
        if packet.version!=self.versions[packet.bank]:raise ValueError('Stale or future credit packet')
        p=self.parameters[packet.bank];g=packet.gradient
        if g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device:raise ValueError('Gradient schema mismatch')
        if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite credit')
        g=g.detach().clone();norm=float(g.norm());cap=self.bank_caps[packet.bank]
        if not math.isfinite(norm):raise ValueError('Bank norm overflow')
        scale=min(1.,cap/(norm+1e-6));g.mul_(scale)
        p.grad=g;optimizer=self.optimizers[packet.bank];optimizer.step();optimizer.zero_grad(set_to_none=True)
        self.versions[packet.bank]+=1
        receipt=dict(bank=packet.bank,issued_version=packet.version,applied_version=self.versions[packet.bank],event_id=packet.event_id,gradient_bytes=g.numel()*g.element_size(),local_clip=cap,raw_norm=norm,scale=scale)
        self.receipts.append(receipt);return receipt

class CalibratedReceiverBankAdamW(ReceiverBankAdamW):
    def __init__(self,model,bank_caps,lr=.003,weight_decay=.01,**options):
        super().__init__(model,lr=lr,weight_decay=weight_decay,**options)
        self.receiver.local=CalibratedLocalCreditBanks(self.receiver.local.parameters.items(),bank_caps,lr=lr,weight_decay=weight_decay,**options)
    def state_dict(self):
        result=super().state_dict();result['bank_caps']=dict(self.receiver.local.bank_caps);return result
    def load_state_dict(self,saved):
        if saved.get('bank_caps')!=self.receiver.local.bank_caps:raise ValueError('Checkpoint changed clipping policy')
        super().load_state_dict(saved)
