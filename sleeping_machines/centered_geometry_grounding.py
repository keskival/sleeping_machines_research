"""TRAIN-normalized local geometry innovations,with unchanged native inference."""
import torch
from torch import nn
from .sparse_geometry_teacher import SparseGroundedNativeComposition,sparse_geometry_targets


class GeometryInnovationScale(nn.Module):
    def __init__(self,levels,width=32):
        super().__init__();self.register_buffer('center',torch.zeros(levels,width));self.register_buffer('scale',torch.ones(levels))

    def set_statistics(self,center,variance):
        center=torch.as_tensor(center,dtype=self.center.dtype,device=self.center.device)
        variance=torch.as_tensor(variance,dtype=self.scale.dtype,device=self.scale.device)
        if center.shape!=self.center.shape or variance.shape!=self.scale.shape or not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(variance).all()) or bool((variance<0).any()):
            raise ValueError('finite TRAIN means and nonnegative level variances required')
        with torch.no_grad():self.center.copy_(center);self.scale.copy_(variance.clamp_min(1e-12).sqrt())

    def forward(self,target,level):return (target-self.center[level])/self.scale[level]


class CenteredGroundedNativeComposition(SparseGroundedNativeComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.normalization=GeometryInnovationScale(self.levels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1009)
            for decoder in self.geometry_decoders:
                nn.init.normal_(decoder.weight,std=.02/self.width**.5);nn.init.zeros_(decoder.bias)

    def load_relational_state(self,weights):
        missing=self.load_state_dict(weights,strict=False)
        if missing.unexpected_keys or any(not k.startswith(('geometry.','geometry_decoders.','normalization.')) for k in missing.missing_keys):
            raise ValueError('native migration coverage changed')
        return missing

    def grounding_loss(self,state,events):
        if state.queue or state.frame_origin is None:raise ValueError('ground only a drained revealed scene')
        rows=torch.as_tensor(events,dtype=torch.float64)
        if len(rows)!=sum(state.input_issued.values()) or float(rows[:,0].max())>state.now:raise ValueError('targets must cover exactly revealed events')
        targets,work=sparse_geometry_targets(rows,self.levels,state.frame_origin,self.geometry.directions)
        losses=[];counts=[];reads=0
        for level,(obj,bank,decoder) in enumerate(zip(self.objects,state.banks,self.geometry_decoders)):
            local=[]
            for address in sorted(bank.published):
                if (level,address) not in targets:continue
                target,_=targets[(level,address)];target=self.normalization(target,level)
                value,_=obj.read(bank,address,state.now);local.append((decoder(value)-target.detach()).square().mean());reads+=1
            counts.append(len(local))
            if local:losses.append(torch.stack(local).mean())
        if not losses:raise ValueError('no observed support')
        state.grounding_work=dict(grounding_reads=reads,grounding_decoder_madds=reads*self.width*32,
            grounding_normalization_elements=reads*32,**work)
        return torch.stack(losses).mean(),dict(level_losses=losses,objects_per_level=counts)
