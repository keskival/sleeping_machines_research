"""Model-wide clipped credit may be delivered to versioned local AdamW banks.

Same gradient-generation/reduction boundary as ordinary AdamW. No local clip,
missing-credit approximation or inference-only bank selection is introduced.
"""
from .receiver_credit_banks import ReceiverCreditBanks

class ReceiverBankAdamW:
    def __init__(self,model,lr=.003,weight_decay=.01,**options):
        self.model=model
        self.receiver=ReceiverCreditBanks(model,lr=lr,weight_decay=weight_decay,**options)
        self.update=0;self.deliveries=0
    def zero_grad(self,set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)
        for opt in self.receiver.local.optimizers.values():opt.zero_grad(set_to_none=set_to_none)
    def step(self):
        # Bounded administrative storage; numerical moments and versions persist.
        self.receiver.local.receipts.clear()
        for packet in self.receiver.packets(f'update{self.update}'):
            self.receiver.apply(packet);self.deliveries+=1
        self.update+=1
    def state_dict(self):
        return dict(banks={n:o.state_dict() for n,o in self.receiver.local.optimizers.items()},
            versions=dict(self.receiver.local.versions),update=self.update,deliveries=self.deliveries)
    def load_state_dict(self,saved):
        if set(saved['banks'])!=set(self.receiver.local.optimizers):raise ValueError('Changed receiver bank layout')
        for name,opt in self.receiver.local.optimizers.items():opt.load_state_dict(saved['banks'][name])
        self.receiver.local.versions=dict(saved['versions']);self.update=saved['update'];self.deliveries=saved['deliveries']
