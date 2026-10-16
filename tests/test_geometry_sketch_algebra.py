import torch
from sleeping_machines.grounded_native_composition import ObservedGeometrySketch
from sleeping_machines.geometry_sketch_algebra import merge_geometry_sketches


def test_exact_observable_child_to_parent_composition():
    teacher=ObservedGeometrySketch().double()
    events=torch.tensor([[0.,2.,3.,1.],[1.,5.,5.,-1.],[2.,3.,7.,1.],[3.,8.,9.,-1.]],dtype=torch.float64)
    origin=(2,3);parent,n=teacher(events,3,(0,0),origin)
    children=[];counts=[];offsets=[]
    for address in [(0,0),(0,1),(1,0),(1,1)]:
        child,count=teacher(events,2,address,origin);children.append(child);counts.append(count)
        offsets.append([4*address[0],4*address[1]])
    merged=merge_geometry_sketches(torch.stack(children),counts,offsets,teacher.directions)
    assert sum(counts)==n and torch.allclose(parent,merged,atol=1e-12,rtol=1e-12)


def test_binding_child_to_role_changes_parent_but_not_child_signature():
    teacher=ObservedGeometrySketch().double()
    a,_=teacher(torch.tensor([[0.,0.,0.,1.]],dtype=torch.float64),2,(0,0),(0,0))
    b,_=teacher(torch.tensor([[0.,1.,1.,-1.]],dtype=torch.float64),2,(0,0),(0,0))
    first=merge_geometry_sketches(torch.stack([a,b]),[1,1],[[0,0],[0,4]],teacher.directions)
    swapped=merge_geometry_sketches(torch.stack([b,a]),[1,1],[[0,0],[0,4]],teacher.directions)
    assert not torch.allclose(first,swapped)


def test_merge_has_live_derivatives_to_nonempty_children_and_zero_empty_case():
    directions=torch.tensor([[0.,0.],[.2,.7]],dtype=torch.float64)
    children=torch.randn(2,8,dtype=torch.float64,requires_grad=True)
    merged=merge_geometry_sketches(children,[2,3],[[0,0],[4,0]],directions)
    gradients=torch.autograd.grad(merged.square().sum(),children)[0]
    assert all(v.norm()>0 for v in gradients)
    assert torch.equal(merge_geometry_sketches(children,[0,0],[[0,0],[4,0]],directions),torch.zeros(8,dtype=torch.float64))
