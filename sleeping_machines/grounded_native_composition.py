"""TRAIN-only observable geometry grounding for native parent representations.

Fixed targets never enter actor state, routing or inference. Their coordinates
come only from revealed sensor events and the actor's observed frame/supports.
"""
import math
import torch
from torch import nn
from .relational_frame_composition import RelationalFrameComposition


class ObservedGeometrySketch(nn.Module):
    def __init__(self,width=32,seed=911):
        super().__init__()
        if width%4:raise ValueError('two polarities, sine/cosine pairs required')
        generator=torch.Generator().manual_seed(seed)
        directions=(torch.rand(width//4,2,generator=generator)*2-1)*math.pi
        directions[0]=0 # explicit signed-count channels, not just oscillations
        self.register_buffer('directions',directions)

    def forward(self,events,level,address,origin):
        events=torch.as_tensor(events,device=self.directions.device,dtype=torch.float64)
        if events.ndim!=2 or events.shape[1]!=4 or not bool(torch.isfinite(events).all()):
            raise ValueError('finite observed time,y,x,polarity rows required')
        if level<0:raise ValueError('nonnegative native support level required')
        scale=2**level
        base=torch.tensor(address,dtype=torch.float64,device=events.device)*scale
        if level:base=base+torch.tensor(origin,dtype=torch.float64,device=events.device)
        xy=events[:,1:3]-base
        keep=((xy>=0)&(xy<scale)).all(1);xy=xy[keep];pol=events[keep,3]
        if not len(xy):return self.directions.new_zeros(self.directions.numel()*2),0
        phase=xy@self.directions.double().T
        features=torch.cat((phase.cos(),phase.sin()),-1)
        channels=[features[pol>0].sum(0),features[pol<=0].sum(0)]
        return (torch.cat(channels)/math.sqrt(len(xy))).to(self.directions.dtype),len(xy)


class GroundedNativeComposition(RelationalFrameComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # This first geometry teacher uses width32; other native widths can use
        # another divisible-by-four sketch size independent of representation.
        self.geometry=ObservedGeometrySketch(32)
        self.geometry_decoders=nn.ModuleList(nn.Linear(self.width,32) for _ in range(self.levels))

    def load_relational_state(self,weights):
        missing=self.load_state_dict(weights,strict=False)
        if missing.unexpected_keys or any(not k.startswith(('geometry.','geometry_decoders.')) for k in missing.missing_keys):
            raise ValueError('native migration coverage changed')
        return missing

    def grounding_loss(self,state,events):
        if state.queue or state.frame_origin is None:raise ValueError('ground only a drained revealed scene')
        observed=torch.as_tensor(events,dtype=torch.float64)
        if len(observed)!=sum(state.input_issued.values()) or float(observed[:,0].max())>state.now:
            raise ValueError('targets must cover exactly the already revealed input events')
        losses=[];counts=[];reads=0;decoder_madds=0;probes=0;target_rows=0
        for level,(obj,bank,decoder) in enumerate(zip(self.objects,state.banks,self.geometry_decoders)):
            local=[]
            for address in sorted(bank.published):
                target,n=self.geometry(events,level,address,state.frame_origin)
                # Geometry targets only train views with actual observed support.
                if not n:continue
                value,_=obj.read(bank,address,state.now);prediction=decoder(value)
                local.append((prediction-target.detach()).square().mean());reads+=1
                decoder_madds+=self.width*32;probes+=n*8;target_rows+=len(events)
            if local:losses.append(torch.stack(local).mean());counts.append(len(local))
            else:counts.append(0)
        if not losses:raise ValueError('no observed support for grounding')
        state.grounding_work=dict(grounding_reads=reads,grounding_decoder_madds=decoder_madds,
            grounding_target_phases=probes,grounding_target_row_tests=target_rows,
            grounding_trig_evaluations=2*probes)
        return torch.stack(losses).mean(),dict(level_losses=losses,objects_per_level=counts)

    def work(self,state):
        result=super().work(state);result.update(getattr(state,'grounding_work',{}));return result
