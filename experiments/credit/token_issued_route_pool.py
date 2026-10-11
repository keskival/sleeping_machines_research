"""Continuous actor issuance with bounded observed-document candidate discovery."""
import torch
from token_issued_route_features import issue
from measure_token_forward_credit import candidates


def issue_with_history(model, observed_tokens, observed_history, initial_state=None,
                       fast_state=None, eos=50256):
    if eos != 50256:
        raise ValueError('This GPT-2 pool uses the recorded EOS protocol')
    if len(observed_history)<len(observed_tokens) or not torch.equal(observed_history[-len(observed_tokens):],observed_tokens):
        raise ValueError('Issued events must be the observed history suffix')
    issued=issue(model,observed_tokens,initial_state,fast_state,eos)
    d=model.embed.embedding_dim;k=model.heads*model.dk
    write=issued['context'][d+k:d+2*k]
    receivers,actions=candidates(observed_history,write,model.heads,model.dk)
    state=issued['carrier'];now=state['position']-1;rows=[]
    with torch.no_grad():
        rates=model.log_rate.detach().exp()
        for r,action in zip(receivers,actions):
            value,when=state['keys'].get(r,(torch.zeros_like(action),now))
            key=value*torch.exp(-rates*(now-when))[:,None]
            scalars=issued['context'].new_tensor([now-when,issued['forecast']['log_probabilities'][0,r].exp(),float((action!=0).any())])
            rows.append(torch.cat([model.embed.weight[r].detach(),key.flatten(),action.flatten(),scalars]))
    return dict(issued,receivers=receivers,actions=actions,descriptors=torch.stack(rows))
