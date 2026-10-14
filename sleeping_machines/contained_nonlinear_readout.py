"""A normalized decoder expansion retaining the learned deep-event prediction."""
import torch
from torch import nn


class ContainedNonlinearHead(nn.Module):
    def __init__(self, base, hidden_width=None, seed=0):
        super().__init__()
        if not isinstance(base,nn.Linear):
            raise ValueError('An existing affine head is required')
        width=base.in_features if hidden_width is None else hidden_width
        if width<1:raise ValueError('Positive lift width required')
        self.base=base
        # Decoder construction must not advance actor/global race randomness.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.lift=nn.Linear(base.in_features,width,device=base.weight.device,dtype=base.weight.dtype)
            self.extra=nn.Linear(width,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
            nn.init.zeros_(self.extra.weight)

    def forward(self, features):
        return self.base(features)+self.extra(torch.tanh(self.lift(features)))

    def added_parameters(self):
        return list(self.lift.parameters())+list(self.extra.parameters())


def expand_adaptive_head(readout, hidden_width=None, seed=0):
    if not readout.adaptive:
        raise ValueError('This candidate expands the frequent-token/branch head only')
    if isinstance(readout.output.head,ContainedNonlinearHead):
        raise ValueError('Do not stack decoder expansions')
    readout.output.head=ContainedNonlinearHead(readout.output.head,hidden_width,seed)
    return readout.output.head


def check_native_readout_containment(readout, features, targets):
    """Frozen native numerical contract; no model fit or optimizer update."""
    import copy
    state=torch.get_rng_state().clone()
    candidate=copy.deepcopy(readout)
    head=expand_adaptive_head(candidate,seed=17014)
    assert torch.equal(state,torch.get_rng_state()),'Expansion changed actor RNG'
    with torch.enable_grad():
        x=features.detach().clone().requires_grad_()
        y=features.detach().clone().requires_grad_()
        baseline=readout.nll(x,targets);expanded=candidate.nll(y,targets)
        assert torch.equal(baseline,expanded),'Initial normalized losses changed'
        gx=torch.autograd.grad(baseline.sum(),x)[0]
        gy,extra_gradient=torch.autograd.grad(expanded.sum(),(y,head.extra.weight))
        assert torch.equal(gx,gy),'Initial factual feature cotangent changed'
    return dict(status='passed',normalized_loss_parity=True,feature_cotangent_parity=True,
        actor_rng_parity=True,scored_targets=2*targets.numel(),
        added_parameters=sum(p.numel() for p in head.added_parameters()),
        initial_extra_weight_gradient_norm=float(extra_gradient.norm()),
        scope='Frozen native readout contract; affine head plus zero-output-weight tanh lift. Prediction containment and factual feature cotangent parity, not fit quality or optimizer-step parity. Extra gradient coordinates can alter global clipping and later Adam updates. No temporal-core shortcut.')
