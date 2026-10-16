import io
import torch
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.packed_queryable_local_deep_race import PackedQueryableLocalDeepRace
from sleeping_machines.packed_local_deep_race import FIELDS,field


def test_packed_queries_publications_and_every_parameter_gradient_match_reference():
    torch.manual_seed(16901)
    reference=QueryableLocalDeepRace(width=8,depth=3,heads=2,pool=3,publication_capacity=2).double()
    packed=PackedQueryableLocalDeepRace(reference);rs=reference.new_state();ps=packed.new_state()
    rg=torch.Generator().manual_seed(11);pg=torch.Generator().manual_seed(11)
    ra=torch.Generator().manual_seed(12);pa=torch.Generator().manual_seed(12)
    rp=[];pp=[];ro=[];po=[]
    for message in torch.randn(4,8,dtype=torch.float64):
        noise=torch.empty(3,2,3,dtype=torch.float64).exponential_();alt=torch.empty_like(noise).exponential_()
        r,rs,p,rm=reference.consume(message,rs,rg,ra,noise=noise,alternative_noise=alt);rp.extend(p);ro.append(r)
        v,ps,p,pm=packed.consume(message,ps,pg,pa,noise=noise,alternative_noise=alt);pp.extend(p);po.append(v)
        assert rm['winners']==pm['winners']
        assert torch.allclose(r,v,atol=2e-11,rtol=2e-11)
        assert abs(rm['ready']-pm['ready'])<2e-12
    target=torch.randn(4,3,8,dtype=torch.float64)
    (torch.stack(ro)*target).sin().mean().backward();reference.deliver_credit(rp,rs)
    (torch.stack(po)*target).sin().mean().backward();packed.deliver_credit(pp,ps)
    for d,layer in enumerate(reference.units):
        for h,units in enumerate(layer):
            for u,unit in enumerate(units):
                for name,path in FIELDS.items():
                    expected=field(unit,path).grad
                    if expected is None:expected=torch.zeros_like(field(unit,path))
                    assert torch.allclose(packed.banks[name].grad[d,h,u],expected,atol=3e-10,rtol=3e-10),(d,h,u,name)
        assert torch.allclose(packed.mix.grad[d],reference.mix[d].weight.grad,atol=3e-10,rtol=3e-10)
        for h in range(2):assert torch.allclose(packed.query.grad[d,h],reference.query[d][h].weight.grad,atol=3e-10,rtol=3e-10)
    for name in ['transport_rate','transport_frequency']:
        assert torch.allclose(getattr(packed,name).grad,getattr(reference,name).grad,atol=3e-10,rtol=3e-10)
    for group in ['memory_queries','memory_keys','assemblers']:
        for (name,p),(other,q) in zip(getattr(packed,group).named_parameters(),getattr(reference,group).named_parameters()):
            assert name==other and p.grad is not None and q.grad is not None
            assert torch.allclose(p.grad,q.grad,atol=3e-10,rtol=3e-10),(group,name)
    for count in ['memory_key_scores','memory_selected_reads','memory_alternative_reads','memory_pair_products','queried_historical_versions']:
        assert ps[count]==rs[count]
    for key,entry in ps['bindings'].entries.items():
        rentry=rs['bindings'].entries[key]
        assert torch.allclose(entry.native.key_view,rentry.native.key_view,atol=2e-11,rtol=2e-11)
        assert all(t.value.grad_fn is None for t in entry.native.receivers)
    snapshot=io.BytesIO();torch.save((ps,pg.get_state(),pa.get_state()),snapshot)
    message=torch.randn(8,dtype=torch.float64)
    expected=packed.consume(message,ps,pg,pa,learn=False)[0]
    snapshot.seek(0);state,gs,als=torch.load(snapshot,weights_only=False);pg.set_state(gs);pa.set_state(als)
    actual=packed.consume(message,state,pg,pa,learn=False)[0]
    assert torch.equal(expected,actual)
