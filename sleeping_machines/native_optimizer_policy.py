"""Semantic AdamW groups for native packed clocks and optional capacity.

Tensor rank is not a parameter's meaning: packed clocks/biases have several
leading bank dimensions. Preserve optimizer moments when regrouping acquired
parameters. This is a new-run policy, never a live-trainer mutation.
"""
import math

def protect_from_decay(name,parameter):
    tail=name.split('.')[-1]
    protected={'raw_rate','frequency','clock_bias','transport_rate','transport_frequency','log_rate','log_beta','freq','age_rep','age_cont','alpha_new','alpha_old','depth_bias','document_bias','bias','gate_b','control_b','unit_bias','word_bias','class_bias','key','key_read','unit_key','unit_out','unit_val','key1','key2'}
    return parameter.ndim<2 or tail in protected

def native_param_groups(model,weight_decay):
    if not math.isfinite(weight_decay) or weight_decay<0:raise ValueError('finite nonnegative weight decay required')
    groups=[dict(params=[],weight_decay=weight_decay),dict(params=[],weight_decay=0.)]
    for name,p in model.named_parameters():groups[int(protect_from_decay(name,p))]['params'].append(p)
    return groups

def regroup_native_optimizer(model,optimizer,weight_decay):
    if not math.isfinite(weight_decay) or weight_decay<0:raise ValueError('finite nonnegative weight decay required')
    names={id(p):n for n,p in model.named_parameters()};seen=[];groups=[]
    # Keep each source group's distinct lr/betas/step settings. Never merge
    # groups merely because their weight decay happens to match.
    for group in optimizer.param_groups:
        halves=[[],[]]
        for parameter in group['params']:
            if id(parameter) not in names:raise ValueError('optimizer parameter is outside this model')
            name=names[id(parameter)];halves[int(protect_from_decay(name,parameter))].append(parameter);seen.append(id(parameter))
        settings={k:v for k,v in group.items() if k not in ['params','weight_decay']}
        groups.extend(dict(settings,params=values,weight_decay=0. if i else weight_decay) for i,values in enumerate(halves) if values)
    if len(set(seen))!=len(seen) or set(seen)!=set(names):raise ValueError('optimizer must partition model parameters exactly')
    optimizer.param_groups=[]
    for group in groups:optimizer.add_param_group(group)
    return dict(groups=len(groups),moments_preserved=True,protected=[n for n,p in model.named_parameters() if protect_from_decay(n,p)],scope='Packed clock/bias/key semantics explicit. Other dormant receiver matrices still incur ordinary AdamW work/decay; selective optimizer execution is not implemented by grouping.')
