import torch
from sleeping_machines.local_deep_race import LocalDeepRace
from sleeping_machines.packed_local_deep_race import PackedLocalDeepRace,FIELDS,field


def test_packed_native_events_and_all_mapped_gradients_match_local_reference():
    torch.manual_seed(916);reference=LocalDeepRace(width=8,depth=3,heads=2,pool=3).double()
    packed=PackedLocalDeepRace(reference);rs=reference.new_state();ps=packed.new_state()
    generator=torch.Generator().manual_seed(914);alternative=torch.Generator().manual_seed(915)
    messages=torch.randn(4,8,dtype=torch.float64);rp=[];pp=[];rout=[];pout=[]
    for message in messages:
        noise=torch.empty(3,2,3,dtype=torch.float64).exponential_();alt=torch.empty_like(noise).exponential_()
        r,rs,packets,rm=reference.consume(message,rs,generator,alternative,noise=noise,alternative_noise=alt);rp.extend(packets)
        p,ps,packets,pm=packed.consume(message,ps,generator,alternative,noise=noise,alternative_noise=alt);pp.extend(packets)
        assert rm['winners']==pm['winners']
        assert abs(rm['ready']-pm['ready'])<1e-12
        assert torch.allclose(r,p,atol=1e-11,rtol=1e-11)
        rout.append(r);pout.append(p)
    target=torch.randn(4,3,8,dtype=torch.float64)
    (torch.stack(rout)*target).sin().mean().backward();reference.deliver_credit(rp,rs)
    (torch.stack(pout)*target).sin().mean().backward();packed.deliver_credit(pp,ps)
    for d,layer in enumerate(reference.units):
        for h,units in enumerate(layer):
            for u,unit in enumerate(units):
                for name,path in FIELDS.items():
                    expected=field(unit,path).grad
                    if expected is None:expected=torch.zeros_like(field(unit,path))
                    actual=packed.banks[name].grad[d,h,u]
                    assert torch.allclose(actual,expected,atol=2e-10,rtol=2e-10),(d,h,u,name,float((actual-expected).abs().max()))
        assert torch.allclose(packed.mix.grad[d],reference.mix[d].weight.grad,atol=2e-10,rtol=2e-10)
        for h in range(2):assert torch.allclose(packed.query.grad[d,h],reference.query[d][h].weight.grad,atol=2e-10,rtol=2e-10)
    assert torch.allclose(packed.transport_rate.grad,reference.transport_rate.grad,atol=2e-10,rtol=2e-10)
    assert torch.allclose(packed.transport_frequency.grad,reference.transport_frequency.grad,atol=2e-10,rtol=2e-10)
    for (d,h,u),trace in rs['receivers'].items():
        assert torch.allclose(ps['layers'][d]['value'][h,u],trace.value,atol=1e-11,rtol=1e-11)
        for name in ['input_real','input_imag','control_weight','control_bias','raw_rate','frequency']:
            assert torch.allclose(ps['layers'][d][name][h,u],getattr(trace,name),atol=1e-11,rtol=1e-11)
    assert packed.storage(ps)['temporal_graph_entries']==0
