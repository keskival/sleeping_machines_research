"""Independent native temporal suffix with protected parent feedback/timing.

Parent service and branch service have distinct availability times. This is
an eager causal event-time implementation, not a physical asynchronous runtime.
Branch output selection/credit policy is deliberately external to this primitive.
"""
import inspect
import torch
from . import sparse_counterfactual_episodes as episodes
from .ordered_token_episodes import ContentFunctions

def protected_branch_features(model,tokens,state=None,generator=None,token_content=None,output='parent',parent_readout='last',pair=False,**kwargs):
    if parent_readout not in ('last','all'):raise ValueError('Declare old readout interface')
    if output not in ('parent','branch'):raise ValueError('Declare parent or branch prediction service')
    if kwargs.get('compiled',False):raise ValueError('Eager native branch primitive')
    if not 0<model.parent_depth<model.depth:raise ValueError('Native grown core required')
    if state is not None and len(state['mem'])<model.depth:
        blank=episodes.initial_state(model,tokens.shape[0]);state=dict(state)
        for field in ('mem','arr','seen'):state[field]=list(state[field])+blank[field][len(state[field]):]
    source=inspect.getsource(episodes.token_features)
    edits=[
      ('    every = []','    every = []\n    branch_ready=st.get("branch_ready", torch.zeros(n,dtype=torch.float64))\n    query_arrivals=[]'),
      ('D - 1, h)','model.parent_depth - 1, h)'),
      ('        for depth in range(D):','''        parent_snapshots=[]
        for depth in range(D):
            if depth == model.parent_depth:
                arrival=torch.maximum(parent_arrival,branch_ready)
                x=torch.cat([transport(parent_x[:,h*P:(h+1)*P],arrival-parent_arrival,model.parent_depth-1,h) for h in range(H)],-1)'''),
      ('exponential_(generator=generator) for _ in range(H)','exponential_(generator=generator if depth < model.parent_depth else model.new_depth_generator) for _ in range(H)'),
      ('            writes[depth] +=', '''            if depth < model.parent_depth:parent_snapshots.append((x,arrival))
            if depth == model.parent_depth-1:
                parent_x,parent_arrival,parent_values,parent_arrivals=x,arrival,values,arrivals
            writes[depth] +='''),
      ('        ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals;', '        branch_ready=arrival\n        ctx_vals = parent_values.reshape(n, H * P); ctx_arr = parent_arrivals;'),
      ('        every.append(x)', '''        parent_features=parent_x
        if parent_readout == "all":
            aligned=[]
            for d,(value,issued_at) in enumerate(parent_snapshots):
                if d==model.parent_depth-1:aligned.append(value)
                else:aligned.append(torch.cat([transport(value[:,h*P:(h+1)*P],parent_arrival-issued_at,d,h) for h in range(H)],-1))
            parent_features=torch.cat(aligned,-1)
        every.append(torch.cat((parent_features,x),-1) if pair else (parent_features if output == "parent" else x))
        query_arrivals.append(parent_arrival if output == "parent" else arrival)'''),
      ('    return torch.stack(every, 1), state, pis','    state["branch_ready"]=branch_ready\n    state["query_arrivals"]=torch.stack(query_arrivals,1)\n    return torch.stack(every, 1), state, pis')]
    for anchor,replacement in edits:
        if source.count(anchor)!=1:raise ValueError('Native branch source contract changed: '+anchor)
        source=source.replace(anchor,replacement)
    namespace=dict(episodes.token_features.__globals__);namespace['output']=output;namespace['parent_readout']=parent_readout;namespace['pair']=pair
    if token_content is not None:namespace['F']=ContentFunctions(tokens,token_content)
    exec(compile(source,'<protected native temporal branch>','exec'),namespace)
    return namespace['token_features'](model,tokens,state,generator,**kwargs)
