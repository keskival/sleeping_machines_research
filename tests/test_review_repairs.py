import copy,io,itertools
import pytest,torch
from torch.nn import functional as F
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.sampled_compositional_credit import SampledCompositionalCredit
from sleeping_machines.direct_input_producer_credit import DirectInputProducerCredit
from sleeping_machines.stable_native_future_credit import StableNativeFutureCredit
from sleeping_machines.producer_replay_native_race import local_write
from experiments.r1_rmr_writer_input_credit import WriterInputCreditRMR
from experiments.r1_rmr import RMR
from tests.test_native_query_migration import identical


def direct_setup():
    torch.manual_seed(17201)
    source=QueryableLocalDeepRace(8,depth=3,heads=2,pool=3,publication_capacity=3).double()
    return SampledCompositionalCredit(source),DirectInputProducerCredit(source)


def oracle(model,state,key,now):
    record=state['bindings'].entries[key].native;p=state['producer_records'][key];d=key[0]
    base=p.base_input;views=dict(p.views)
    edge=state['direct_producer_edges'].get(key)
    if edge is not None:base=oracle(model,state,edge.source,edge.issued)
    lineage=state['compositional_lineage'].get(key)
    if lineage is not None:
        for role,child in enumerate(lineage.sources):views[role]=model.aligned_child(d,oracle(model,state,child,lineage.issued[role]),lineage,role)
    stream=base+p.inbox
    if views:stream=stream+model.assemblers[d-1](views)
    incoming=F.linear(stream,model.mix[d]).reshape(model.heads,model.payload)
    memory=local_write(p.prior_value,p.write_parameters,incoming,p.elapsed)
    hi=torch.arange(model.heads);winner=torch.tensor(record.winners);params={n:b[d,hi,winner] for n,b in model.banks.items()}
    value=model.unit_value(params,incoming,memory);age=now-record.head_times
    mag=torch.exp(-age[:,None]*(F.softplus(model.transport_rate[d])+1e-6));angle=age[:,None]*model.transport_frequency[d]
    pair=value.reshape(model.heads,-1,2);re,im=pair[...,0]*mag,pair[...,1]*mag
    aligned=torch.stack((re*angle.cos()-im*angle.sin(),re*angle.sin()+im*angle.cos()),-1).flatten()
    return F.layer_norm(stream+aligned/model.depth**.5,(model.width,))


def paths(model,state,key):
    edges=model.input_edges(state,key)
    if not edges:return [((),1.)]
    result=[]
    for i,(kind,item) in enumerate(edges):
        child=item.source if kind=='direct' else state['compositional_lineage'][key].sources[item]
        result += [((i,)+suffix,p/len(edges)) for suffix,p in paths(model,state,child)]
    return result


def test_direct_and_query_path_expectation_matches_complete_input_dag():
    _,m=direct_setup();state=m.new_state();g=torch.Generator().manual_seed(31);a=torch.Generator().manual_seed(32)
    for x in torch.randn(5,8,dtype=torch.float64):m.consume(x,state,g,a)
    key=state['publications'][2][-1];now=state['bindings'].entries[key].available+.1;probe=torch.randn(8,dtype=torch.float64)
    expected_value=oracle(m,state,key,now);(expected_value*probe).sum().backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters() if n=='mix' or n.startswith('assemblers.') if p.grad is not None}
    average={n:torch.zeros_like(p) for n,p in expected.items()}
    for path,probability in paths(m,state,key):
        iterator=iter(path);m.select_credit_role=lambda state,roles:(next(iterator),1/len(roles))
        m.zero_grad(set_to_none=True);actual=m.read_publication(state,key,now,[],learn=True)
        torch.testing.assert_close(actual,expected_value,rtol=1e-9,atol=1e-9)
        (actual*probe).sum().backward()
        for n,p in m.named_parameters():
            if n in average and p.grad is not None:average[n]+=probability*p.grad
    for n,value in expected.items():torch.testing.assert_close(average[n],value,atol=3e-8,rtol=3e-8)
    assert average['mix'][0].norm()>0


def test_direct_credit_forward_rng_gc_and_full_state_resume():
    old,m=direct_setup();s=m.new_state();os=old.new_state();g=torch.Generator().manual_seed(31);a=torch.Generator().manual_seed(32);og=torch.Generator().manual_seed(31);oa=torch.Generator().manual_seed(32)
    for x in torch.randn(12,8,dtype=torch.float64):
        y,os,_,_=old.consume(x,os,og,oa);z,s,_,_=m.consume(x,s,g,a)
        assert torch.equal(y,z) and torch.equal(g.get_state(),og.get_state()) and torch.equal(a.get_state(),oa.get_state())
    assert all(edge.source in s['bindings'].entries for edge in s['direct_producer_edges'].values())
    assert len(s['bindings'].entries)<120
    saved=io.BytesIO();torch.save((m.state_dict(),s,g.get_state(),a.get_state()),saved);x=torch.randn(8,dtype=torch.float64)
    expected=m.consume(x,s,g,a);saved.seek(0);weights,s,gs,als=torch.load(saved,weights_only=False)
    m.load_state_dict(weights);g.set_state(gs);a.set_state(als);actual=m.consume(x,s,g,a)
    assert torch.equal(expected[0],actual[0]) and identical(expected[1],actual[1])


def test_fixed_feature_epoch_ridge_forgetting_and_recalibration():
    torch.manual_seed(17202);m=StableNativeFutureCredit(8,depth=3,heads=2,pool=3,forgetting=.97)
    s=m.new_state();g=torch.Generator().manual_seed(33);a=torch.Generator().manual_seed(34)
    params=copy.deepcopy(dict(m.named_parameters()));rows=[];targets=[];vectors=[torch.randn(8) for _ in range(3)]
    for k in range(8):
        m.begin_decision();q,_,packets=m.quote(*vectors,.5,.001,s,g,a);j=0 if k%2 else 2
        feature=torch.cat(m.features)[j].detach().double();target=feature@torch.arange(len(feature),dtype=torch.float64)/10
        rows.append(feature);targets.append(target);m.learn_audit(q,j,target,packets,s)
    for n,p in m.named_parameters():assert torch.equal(params[n],p)
    X=torch.stack(rows);Y=torch.stack(targets);weights=.97**torch.arange(7,-1,-1,dtype=torch.float64)
    information=.1*.97**8*torch.eye(X.shape[1],dtype=torch.float64)+X.T@(weights[:,None]*X)
    expected=torch.linalg.solve(information,X.T@(weights*Y))
    torch.testing.assert_close(m.response,expected,atol=1e-10,rtol=1e-10)
    with torch.no_grad():m.context.weight.add_(.01)
    with pytest.raises(RuntimeError):m.begin_decision()
    m.recalibrate();assert int(m.feature_epoch)==1 and int(m.observations)==0 and not m.response.any()
    with pytest.raises(ValueError):m.choose_service(torch.tensor([float('nan')]),uncertainty_price=.5)


def test_writer_input_value_is_unchanged_and_first_order_path_is_live():
    # The stopped old-correction contribution has a first-order VJP. A leaf
    # write isolates it from unrelated read/parameter paths.
    base=torch.randn(1,4,8,dtype=torch.float64,requires_grad=True);write=torch.randn(1,4,8,dtype=torch.float64,requires_grad=True)
    idx=torch.tensor([[[0],[0],[1],[2]]]);weight=torch.ones(1,4,1,dtype=torch.float64)
    selected=torch.tensor([[[0],[1]]]);rows=torch.tensor([2,3]);writes=[(write,idx,weight)]
    reads=base.new_zeros(1,2,4).scatter_(2,selected.reshape(1,2,1),1.)
    with torch.no_grad():old=RMR.corr_values(reads,writes,rows).to(base.dtype)
    stopped=(base.gather(1,selected.reshape(1,2,1).expand(-1,-1,8))+old).reshape(1,2,1,8)
    # Explicit old stopped reference; production RMR now also restores this VJP.
    live=WriterInputCreditRMR.slot_values(object.__new__(WriterInputCreditRMR),base,writes,selected,rows)
    assert torch.equal(stopped,live)
    assert torch.autograd.grad(stopped.sum(),write,allow_unused=True)[0] is None
    gradient=torch.autograd.grad(live.sum(),write)[0]
    assert gradient.norm()>0 and torch.isfinite(gradient).all()


def test_semantic_optimizer_groups_protect_banked_clocks_and_preserve_moments():
    from sleeping_machines.native_optimizer_policy import regroup_native_optimizer,protect_from_decay
    from sleeping_machines.local_deep_race import LocalDeepRace
    from sleeping_machines.packed_local_deep_race import PackedLocalDeepRace
    m=PackedLocalDeepRace(LocalDeepRace(8,depth=3,heads=2,pool=3));opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    for p in m.parameters():p.grad=torch.ones_like(p)
    opt.step();before={p:copy.deepcopy(opt.state[p]) for p in m.parameters()}
    receipt=regroup_native_optimizer(m,opt,.01)
    assert receipt['moments_preserved']
    for p,value in before.items():assert identical(value,opt.state[p])
    named=dict(m.named_parameters());rates=named['banks.raw_rate'];bias=named['banks.control_b']
    assert rates.ndim>1 and bias.ndim>1 and protect_from_decay('banks.raw_rate',rates) and protect_from_decay('banks.control_b',bias)
    for group in opt.param_groups:
        for p in group['params']:
            name=next(n for n,v in named.items() if v is p)
            assert group['weight_decay']==(0. if protect_from_decay(name,p) else .01)


def test_factual_direct_message_and_current_write_input_vjp_matches_finite_difference():
    # No discrete ambiguity: one receiver/head and first event. The complete
    # current-event conditional input derivative must match the actual forward.
    torch.manual_seed(17401)
    m=DirectInputProducerCredit(QueryableLocalDeepRace(8,depth=2,heads=1,pool=1,publication_capacity=3).double())
    x=torch.randn(8,dtype=torch.float64);probe=torch.randn(8,dtype=torch.float64)
    def forward(learn):
        h,s,_,_=m.consume(x,m.new_state(),torch.Generator().manual_seed(77),torch.Generator().manual_seed(78),learn=learn)
        return h[-1]@probe,s
    objective,state=forward(True);gradient=torch.autograd.grad(objective,m.mix)[0][0]
    original=m.mix.detach().clone();direction=torch.randn_like(original[0]);epsilon=1e-5
    with torch.no_grad():m.mix[0].copy_(original[0]+epsilon*direction)
    plus,_=forward(False)
    with torch.no_grad():m.mix[0].copy_(original[0]-epsilon*direction)
    minus,_=forward(False)
    with torch.no_grad():m.mix.copy_(original)
    finite=float((plus-minus).detach())/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(),torch.tensor(finite,dtype=torch.float64),atol=2e-7,rtol=2e-6)
    assert state['immediate_direct_replays']==1 and state['current_write_input_vjps']==2


def test_checkpointed_writer_vjp_matches_live_first_order_gradient():
    from experiments.r1_rmr_writer_input_credit import CheckpointedWriterInputCreditRMR
    torch.manual_seed(17402)
    base=torch.randn(1,5,8,requires_grad=True);write=torch.randn(1,5,2,8,requires_grad=True)
    idx=torch.tensor([[[0,0],[0,1],[1,0],[2,1],[3,2]]]);weight=torch.rand(1,5,2,requires_grad=True);beta=torch.tensor(.1,requires_grad=True)
    selected=torch.tensor([[[0,1],[1,2]]]);rows=torch.tensor([3,4]);args=(base,[(write,idx,weight,beta)],selected,rows)
    full=WriterInputCreditRMR.slot_values(object.__new__(WriterInputCreditRMR),*args)
    expected=torch.autograd.grad(full.square().sum(),(base,write,weight,beta))
    result=CheckpointedWriterInputCreditRMR.slot_values(object.__new__(CheckpointedWriterInputCreditRMR),*args)
    actual=torch.autograd.grad(result.square().sum(),(base,write,weight,beta))
    assert torch.equal(full,result)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-6)


def test_zero_effect_interventions_have_zero_values_and_prefer_noop():
    m=StableNativeFutureCredit(8,depth=2,heads=2,pool=2);s=m.new_state();m.begin_decision()
    q,_,_=m.quote(*(torch.randn(8) for _ in range(3)),.5,.001,s,torch.Generator().manual_seed(1),torch.Generator().manual_seed(2),actual_action_norm=0.)
    assert torch.equal(q,torch.zeros_like(q)) and not torch.cat(m.features).any()
    assert m.choose_service(q)==(1,0.)


def test_invalid_actual_action_norm_does_not_mutate_critic_state():
    m=StableNativeFutureCredit(8,depth=2,heads=2,pool=2);state=m.new_state();before=copy.deepcopy(state)
    with pytest.raises(ValueError):m.quote(*(torch.randn(8) for _ in range(3)),.5,.001,state,torch.Generator().manual_seed(1),torch.Generator().manual_seed(2),actual_action_norm=-1.)
    assert identical(state,before) and not m.features


def test_fixed_feature_critic_restores_into_a_fresh_constructor():
    torch.manual_seed(17403);m=StableNativeFutureCredit(8,depth=3,heads=2,pool=3,forgetting=.99)
    s=m.new_state();g=torch.Generator().manual_seed(81);a=torch.Generator().manual_seed(82);vectors=[torch.randn(8) for _ in range(3)]
    def step(model,state,generator,alternative):
        model.begin_decision();q,_,packets=model.quote(*vectors,.5,.001,state,generator,alternative)
        return q.detach(),model.learn_audit(q,0,-.002,packets,state)
    step(m,s,g,a);saved=io.BytesIO();torch.save((m.state_dict(),s,g.get_state(),a.get_state()),saved)
    expected=step(m,s,g,a);final=copy.deepcopy((m.state_dict(),s,g.get_state(),a.get_state()))
    saved.seek(0);weights,state,gs,als=torch.load(saved,weights_only=False)
    rebuilt=StableNativeFutureCredit(8,depth=3,heads=2,pool=3,forgetting=1.);rebuilt.load_state_dict(weights)
    rg=torch.Generator().set_state(gs);ra=torch.Generator().set_state(als);actual=step(rebuilt,state,rg,ra)
    assert torch.equal(actual[0],expected[0]) and actual[1]==expected[1]
    assert identical(final,(rebuilt.state_dict(),state,rg.get_state(),ra.get_state()))


def test_production_accumulated_writer_values_and_gradients_match_reference():
    torch.manual_seed(17404)
    for per_target in [False,True]:
        base=torch.randn(1,8,8,requires_grad=True)
        write=torch.randn((1,8,2,8) if per_target else (1,8,8),requires_grad=True)
        idx=torch.randint(0,8,(1,8,2));weight=torch.rand(1,8,2,requires_grad=True);beta=torch.tensor(.1,requires_grad=True)
        chosen=torch.randint(0,8,(1,4,2));rows=torch.arange(4,8);args=(base,[(write,idx,weight,beta)],chosen,rows)
        reference=WriterInputCreditRMR.slot_values(object.__new__(WriterInputCreditRMR),*args)
        expected=torch.autograd.grad(reference.square().sum(),(base,write,weight,beta))
        fast=RMR.slot_values(object.__new__(RMR),*args)
        actual=torch.autograd.grad(fast.square().sum(),(base,write,weight,beta))
        torch.testing.assert_close(fast,reference,rtol=2e-6,atol=2e-6)
        for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=5e-6,atol=5e-6)


def test_event_slot_memory_exports_exact_persistent_carry():
    from experiments.native_slot_memory import SlotMemory
    torch.manual_seed(17405)
    for product in [False, True]:
        m=SlotMemory(8,16,3,8,explore=2,read_topk=4,product_keys=product).eval()
        u=torch.randn(2,35,8)
        with torch.no_grad():
            expected=m(u); final=tuple(x.clone() for x in m.last_state)
            first,traffic=m.event_forward(u[:,:13]);carry=m.last_state
            second,_=m.event_forward(u[:,13:],carry)
            torch.testing.assert_close(torch.cat([first,second],1),expected,rtol=1e-5,atol=1e-6)
            for x,y in zip(m.last_state,final):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
            assert traffic['carry_materialization_rows_per_lane']==32
