"""Native typed receipts with bounded, rotating addressed clock residuals."""
import copy
import torch
from native_optimizer_receipts import ReceiptCarry,events

def support(version):
    # Two decisions per training receipt: both use the same phase so every
    # mode receives current-decision teaching across sixteen receipts.
    phase=(version//2)%16
    mask=torch.zeros(64,dtype=torch.float64)
    for leg in range(4):mask[16*leg+phase]=1.
    return mask

class AddressedClockCredit(torch.nn.Module):
    def __init__(self,base,scale):
        super().__init__();self.parent=copy.deepcopy(base);self.parent.requires_grad_(False)
        self.receipt=copy.deepcopy(base)
        with torch.random.fork_rng(devices=[]):
            self.receipt.core.head=torch.nn.Linear(self.receipt.core.total_payload,64).double()
        self.core=self.receipt.core
        torch.nn.init.zeros_(self.core.head.weight);torch.nn.init.zeros_(self.core.head.bias)
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float64))
        self.last_parent=None;self.last_mask=None
    def forward(self,features,observed,carry,operator):
        parent_carry=None if carry is None else carry.parent
        receipt_carry=None if carry is None else carry.receipt
        version=0 if parent_carry is None else parent_carry.version
        self.parent.core.train(False)
        with torch.no_grad():baseline,parent_carry=self.parent(features,observed,parent_carry)
        tokens=events(features,observed,operator,baseline[0].detach(),self.scale)
        rng=torch.Generator().manual_seed(99173).get_state() if carry is None else carry.rng
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng);residual,receipt_carry=self.receipt(features,tokens,receipt_carry)
            next_rng=torch.get_rng_state().clone()
        self.last_parent=baseline[0].detach().repeat_interleave(16)
        self.last_mask=support(version).to(features.device)
        return self.last_parent[None]+.5*residual.tanh()*self.last_mask,ReceiptCarry(parent_carry,receipt_carry,next_rng)

def guided(model,grads,credit):
    names=[n for n,_ in model.named_parameters()];out=[g.detach() for g in grads]
    if credit.numel()==4:
        credit=torch.cat([credit[2*j+leg].expand(layer.n)/(layer.n**.5) for j,layer in enumerate(model.layers) for leg in range(2)])
    offset=0
    for j,layer in enumerate(model.layers):
        for key in ('log_rate','freq'):
            i=names.index(f'layers.{j}.{key}');out[i]=out[i]+credit[offset:offset+layer.n];offset+=layer.n
    assert offset==credit.numel()
    return out
