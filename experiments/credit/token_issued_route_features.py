"""Prefix-only actor descriptors, captured before any continuation is read."""
import torch
from token_episode_state_vectorized import consume
from measure_token_forward_credit import candidates


def issue(model, observed_tokens, initial_state=None, fast_state=None, eos=50256):
    if observed_tokens.ndim != 1 or not len(observed_tokens):
        raise ValueError('Observed prefix required')
    captured = {}
    hooks = [model.out_norm.register_forward_hook(lambda m,a,r: captured.update(h=r.detach())),
             model.query_w.register_forward_hook(lambda m,a,r: captured.update(q=r.detach())),
             model.key_w.register_forward_hook(lambda m,a,r: captured.update(w=r.detach()))]
    state = initial_state
    reset_fast = False
    try:
        with torch.no_grad():
            for token in observed_tokens:
                if int(token) == eos: state = None; reset_fast = True
                forecast,state = consume(model, token.reshape(1), state)
    finally:
        for hook in hooks: hook.remove()
    receivers, actions = candidates(observed_tokens, captured['w'], model.heads, model.dk)
    with torch.no_grad():
        z = torch.zeros_like(captured['h'][0]) if fast_state is None or reset_fast else fast_state.detach()
        # h, q, produced message and lexical state are common issued context.
        context = torch.cat([captured['h'].flatten(), captured['q'].flatten(),
                             captured['w'].flatten(), z])
        descriptors=[];now=state['position']-1;rates=model.log_rate.detach().exp()
        for receiver, action in zip(receivers,actions):
            value,when=state['keys'].get(receiver,(torch.zeros_like(action),now))
            key=value*torch.exp(-rates*(now-when))[:,None]
            scalars=context.new_tensor([now-when, forecast['log_probabilities'][0,receiver].exp(),
                                        float((action != 0).any())])
            descriptors.append(torch.cat([model.embed.weight[receiver].detach(),
                                           key.flatten(),action.flatten(),scalars]))
    return dict(context=context, descriptors=torch.stack(descriptors), receivers=receivers,
                actions=actions, carrier=state, forecast=forecast)


class RouteController(torch.nn.Module):
    """One connected scorer shared across all issued optional routes."""
    def __init__(self, context_size, descriptor_size, width=64):
        super().__init__()
        self.context=torch.nn.Linear(context_size,width)
        self.route=torch.nn.Linear(descriptor_size,width)
        self.output=torch.nn.Linear(width,1)
    def forward(self, context, descriptors):
        return self.output(torch.tanh(self.context(context)[None]+self.route(descriptors))).flatten()
