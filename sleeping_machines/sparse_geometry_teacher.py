"""Exact static geometry targets from sparse counts and cached role rotations.

Training-only support code. No model-state shortcut, hidden labels or inference
input is introduced. Fixed signed-polarity counts are sufficient for this
particular static teacher; they are not sufficient for general temporal tasks.
"""
import math
import torch
from .grounded_native_composition import GroundedNativeComposition


def sparse_geometry_targets(events,levels,origin,directions):
    if levels<1 or directions.ndim!=2 or directions.shape[1]!=2:
        raise ValueError('positive levels and two-dimensional Fourier directions required')
    rows=torch.as_tensor(events,dtype=torch.float64,device=directions.device)
    if rows.ndim!=2 or rows.shape[1]!=4 or not bool(torch.isfinite(rows).all()):
        raise ValueError('finite observed time,y,x,polarity rows required')
    if not torch.equal(rows[:,1:3],rows[:,1:3].round()):raise ValueError('integer sensor pixels required')
    if len(origin)!=2 or any(int(x)!=x for x in origin):raise ValueError('integer observed frame origin required')
    # Aggregate repeated observations once; counts retain the static target's
    # polarity field without retaining an event log or evaluating any phase.
    counts={}
    for row in rows:
        xy=tuple(map(int,row[1:3]));value=counts.setdefault(xy,[0,0]);value[0 if row[3]>0 else 1]+=1
    f=len(directions);dtype=directions.dtype;targets={};current={}
    for address,polarity in counts.items():
        signature=torch.zeros(2,2,f,dtype=torch.float64,device=directions.device)
        signature[0,0]=polarity[0];signature[1,0]=polarity[1]
        current[address]=(signature,sum(polarity))
        targets[(0,address)]=((signature.flatten()/math.sqrt(sum(polarity))).to(dtype),sum(polarity))
    edges=0
    for level in range(1,levels):
        half=2**(level-1);offsets=directions.new_tensor([[0,0],[0,half],[half,0],[half,half]],dtype=torch.float64)
        phase=offsets@directions.double().T;cs=phase.cos();ss=phase.sin();parents={}
        for address,(signature,n) in current.items():
            local=tuple(address[i]-int(origin[i]) for i in range(2)) if level==1 else address
            parent=tuple(x//2 for x in local);role=2*(local[0]%2)+(local[1]%2)
            c=signature[:,0];s=signature[:,1]
            rotated=torch.stack((c*cs[role]-s*ss[role],s*cs[role]+c*ss[role]),1)
            if parent in parents:
                old,total=parents[parent];parents[parent]=(old+rotated,total+n)
            else:parents[parent]=(rotated,n)
            edges+=1
        current=parents
        for address,(signature,n) in current.items():
            targets[(level,address)]=((signature.flatten()/math.sqrt(n)).to(dtype),n)
    work=dict(sparse_teacher_observation_visits=len(rows),sparse_teacher_pixel_records=len(counts),
       sparse_teacher_target_records=len(targets),sparse_teacher_merge_edges=edges,
       sparse_teacher_phase_evaluations=4*max(0,levels-1)*f,
       sparse_teacher_trig_evaluations=8*max(0,levels-1)*f,
       sparse_teacher_rotation_products=edges*8*f,
       sparse_teacher_target_tensor_bytes=sum(v.numel()*v.element_size() for v,n in targets.values()))
    return targets,work


class SparseGroundedNativeComposition(GroundedNativeComposition):
    def grounding_loss(self,state,events):
        if state.queue or state.frame_origin is None:raise ValueError('ground only a drained revealed scene')
        rows=torch.as_tensor(events,dtype=torch.float64)
        if len(rows)!=sum(state.input_issued.values()) or float(rows[:,0].max())>state.now:
            raise ValueError('targets must cover exactly revealed events')
        targets,work=sparse_geometry_targets(rows,self.levels,state.frame_origin,self.geometry.directions)
        losses=[];counts=[];reads=0
        for level,(obj,bank,decoder) in enumerate(zip(self.objects,state.banks,self.geometry_decoders)):
            local=[]
            for address in sorted(bank.published):
                if (level,address) not in targets:continue
                target,_=targets[(level,address)];value,_=obj.read(bank,address,state.now)
                local.append((decoder(value)-target.detach()).square().mean());reads+=1
            counts.append(len(local))
            if local:losses.append(torch.stack(local).mean())
        if not losses:raise ValueError('no observed support')
        state.grounding_work=dict(grounding_reads=reads,grounding_decoder_madds=reads*self.width*32,**work)
        return torch.stack(losses).mean(),dict(level_losses=losses,objects_per_level=counts)
