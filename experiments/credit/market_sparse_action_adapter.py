"""Use the contracted sparse optimizer metric in native utility acquisition."""
import market_actual_update_teacher as teacher
from market_sparse_action_metric import sparse_adam_metric
from market_action_geometry import teaching_direction
torch,np=teacher.torch,teacher.np

def parameter_metric(model,opt,grads,credit):
    parameters=list(model.named_parameters());issued=teacher.guided(model,grads,credit)
    flat=torch.cat([g.detach().reshape(-1) for g in issued]);first=[];second=[];steps=[];offsets={};offset=0
    for name,p in parameters:
        offsets[name]=offset;offset+=p.numel();state=opt.state[p]
        first.append(state.get('exp_avg',torch.zeros_like(p)).reshape(-1));second.append(state.get('exp_avg_sq',torch.zeros_like(p)).reshape(-1));steps.append(int(state.get('step',0))+1)
    assert len(set(steps))==1
    supports=[]
    for j,layer in enumerate(model.layers):
        for key in ('log_rate','freq'):
            offset=offsets[f'layers.{j}.{key}'];indices=torch.arange(offset,offset+layer.n)
            supports.append((indices,flat.new_full((layer.n,),1/np.sqrt(layer.n))))
    group=opt.param_groups[0];beta1,beta2=group['betas']
    return sparse_adam_metric(flat,torch.cat(first),torch.cat(second),supports,step=steps[0],lr=group['lr'],beta1=beta1,beta2=beta2,adam_eps=group['eps'])
