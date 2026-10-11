"""Source-preserving packet construction using vectorized native key transport."""
import torch
from token_forward_credit import TokenPacket
from token_episode_state_vectorized import consume

def packet(model,tokens,index=3,initial_state=None,eos_token=50256):
    if tokens.ndim!=1 or len(tokens)<index+2 or not 0<=index<len(tokens)-1:
        raise ValueError('One observed token stream and at least one post-write target required')
    if not model.keyed:raise ValueError('Keyed token model required')
    probabilities=[];queries=[];resets=[];live=[];state=initial_state;active=False
    captured=[];hook=model.query_w.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
    try:
        with torch.no_grad():
            for j in range(len(tokens)-1):
                reset=int(tokens[j])==eos_token
                if reset:state=None;active=False
                forecast,state=consume(model,tokens[j:j+1],state,parameter_version=0)
                if j==index:active=True
                if j>=index:
                    probabilities.append(forecast['log_probabilities'][0]);queries.append(captured[-1].reshape(model.heads,model.dk))
                    resets.append(reset);live.append(active)
    finally:hook.remove()
    return TokenPacket(torch.stack(probabilities),torch.stack(queries),tokens[index+1:],torch.tensor(resets),torch.tensor(live),model.embed.weight.detach(),model.log_rate.detach().exp(),model.dk,index,state,int(tokens[-1])==eos_token)

