"""Exact role-aware composition law for the fixed observed geometry teacher.

Audit/support algebra, not an inference bypass or a semantic sufficiency claim.
"""
import torch


def merge_geometry_sketches(sketches,counts,offsets,directions):
    """Merge child-relative signatures into a parent-relative signature.

Offsets are child_origin - parent_origin in observed pixel coordinates. Each
signature is polarity-separated [cosines,sines], normalized by sqrt(child N).
"""
    if sketches.ndim!=2 or directions.ndim!=2 or directions.shape[1]!=2:
        raise ValueError('matrix signatures and two-dimensional directions required')
    k=sketches.shape[0];f=directions.shape[0]
    counts=torch.as_tensor(counts,dtype=sketches.dtype,device=sketches.device)
    offsets=torch.as_tensor(offsets,dtype=sketches.dtype,device=sketches.device)
    if sketches.shape[1]!=4*f or counts.shape!=(k,) or offsets.shape!=(k,2) or bool((counts<0).any()):
        raise ValueError('compatible signatures,nonnegative counts and geometric offsets required')
    if not all(bool(torch.isfinite(x).all()) for x in [sketches,counts,offsets,directions]):
        raise ValueError('finite geometric records required')
    if float(counts.sum())==0:return sketches.new_zeros(4*f)
    phase=offsets@directions.to(sketches).T;c=phase.cos();s=phase.sin()
    roles=sketches.reshape(k,2,2,f);cosine=roles[:,:,0];sine=roles[:,:,1]
    rotated=torch.stack((cosine*c[:,None]-sine*s[:,None],sine*c[:,None]+cosine*s[:,None]),2)
    weights=(counts/counts.sum()).sqrt()
    return (rotated*weights[:,None,None,None]).sum(0).flatten()
