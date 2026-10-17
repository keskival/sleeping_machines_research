"""Continue persistent causal state onto a declared new observation tape.

The last revealed target is the next input. At a tape boundary it forecasts
 the first new target; it must not be silently replaced by that new target.
"""
import numpy as np
import torch

def transition_batch(data,start,count,state,*,tape_start):
    if count<=0 or start<tape_start or start+count>len(data):
        raise ValueError('bounded forward tape slice required')
    expected=state.get('expected_next_input')
    if expected is None:raise ValueError('persistent predecessor required')
    if start>tape_start and int(data[start-1])!=expected:
        raise ValueError('new tape must continue on its revealed predecessor')
    targets=torch.from_numpy(np.asarray(data[start:start+count]).astype(np.int64))
    return torch.cat([targets.new_tensor([expected]),targets])
