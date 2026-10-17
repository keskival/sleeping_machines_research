import copy,itertools,io
import pytest,torch
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace
from sleeping_machines.direct_input_producer_credit import DirectInputProducerCredit
from sleeping_machines.outcome_compositional_credit import OutcomeCompositionalCredit
from sleeping_machines.coupled_native_query_credit import ordered_pair_probabilities
from sleeping_machines.native_indexed_cause_head import indexed_cause_nll
from sleeping_machines.native_cause_bank import NativeCauseBank
from experiments.r1_local_native_deep import LocalNativeDeepLM
from tests.test_native_query_migration import identical


def setup():
    torch.manual_seed(17301);source=QueryableLocalDeepRace(8,depth=2,heads=2,pool=2,publication_capacity=3).double()
    return DirectInputProducerCredit(source),OutcomeCompositionalCredit(source)


def test_issued_forecast_and_factual_rng_match_previous_native_model():
    old,m=setup();os=old.new_state();state=m.new_state();og=torch.Generator().manual_seed(51);oa=torch.Generator().manual_seed(52);g=torch.Generator().manual_seed(51);a=torch.Generator().manual_seed(52)
    for message in torch.randn(5,8,dtype=torch.float64):
        expected,os,_,_=old.consume(message,os,og,oa)
        (actual,state,packets,metadata),pending=m.consume_for_outcome(message,state,g,a)
        assert torch.equal(expected,actual) and torch.equal(g.get_state(),og.get_state()) and torch.equal(a.get_state(),oa.get_state())
        before=copy.deepcopy(state);gs=g.get_state();als=a.get_state();rng=torch.get_rng_state()
        loss=actual[-1].square().sum()
        term,rows=m.outcome_credit(pending,loss,lambda h,s,md:h[-1].square().sum(),state)
        assert float(term.detach())==0 and torch.equal(rng,torch.get_rng_state())
        before['nonlinear_route_audits']=state['nonlinear_route_audits'];before['nonlinear_route_replay_events']=state['nonlinear_route_replay_events']
        assert identical(before,state) and torch.equal(gs,g.get_state()) and torch.equal(als,a.get_state())
        assert all(k in state['direct_producer_edges'] for k in state['compositional_lineage'])
        with pytest.raises(RuntimeError):m.outcome_credit(pending,loss,lambda h,s,md:loss,state)


def test_expected_actual_suffix_route_credit_matches_exact_categorical_gradient():
    _,m=setup();state=m.new_state();g=torch.Generator().manual_seed(53);a=torch.Generator().manual_seed(54)
    for message in torch.randn(2,8,dtype=torch.float64):
        with torch.no_grad():m.consume(message,state,g,a,learn=False)
    message=torch.randn(8,dtype=torch.float64);probe=torch.randn(8,dtype=torch.float64);parameter=m.memory_queries[0].weight
    pairs=list(itertools.permutations(range(3),2));hazards=(torch.tensor(1.1,dtype=torch.float64),torch.tensor(.9,dtype=torch.float64))
    losses={};score_graph=None;pi_values=None
    for pair in pairs:
        m._forced_pairs={1:(pair,hazards)}
        (h,s,_,_),pending=m.consume_for_outcome(message,copy.deepcopy(state),torch.Generator().manual_seed(55),torch.Generator().manual_seed(56))
        losses[pair]=float(((h[-1]@probe).square()-1).square().detach())
        score_graph=pending.decisions[0].scores;pi_values=ordered_pair_probabilities(score_graph).detach()
    expected_objective=sum(ordered_pair_probabilities(score_graph)[pair]*losses[pair] for pair in pairs)
    expected=torch.autograd.grad(expected_objective,parameter)[0];got=torch.zeros_like(expected)
    for factual in pairs:
        for alternative in pairs:
            if alternative==factual:continue
            m._forced_pairs={1:(factual,hazards)}
            (h,s,_,_),pending=m.consume_for_outcome(message,copy.deepcopy(state),torch.Generator().manual_seed(55),torch.Generator().manual_seed(56))
            decision=pending.decisions[0];decision.alternative=alternative
            q=float(pi_values[alternative]/(1-pi_values[factual]));decision.proposal=q
            term,rows=m.outcome_credit(pending,losses[factual],lambda views,s,md:((views[-1]@probe).square()-1).square(),s)
            assert abs(rows[0]['alternative_loss']-losses[alternative])<1e-10
            got+=float(pi_values[factual])*q*torch.autograd.grad(term,parameter)[0]
    torch.testing.assert_close(got,expected,rtol=1e-7,atol=1e-8)
    assert expected.norm()>1e-8


def test_complete_native_actor_nonlinear_task_head_and_optimizer_recovery():
    torch.manual_seed(17302);classes=torch.tensor([0,0,1,1,2,2,3,3]);words=torch.arange(8).reshape(4,2)
    actor=LocalNativeDeepLM(8,classes,words,width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,explore=1,dk=4,packed=True).double()
    actor.core=OutcomeCompositionalCredit(QueryableLocalDeepRace(8,depth=3,heads=2,pool=3,publication_capacity=3).double())
    state=actor.new_state();g=torch.Generator().manual_seed(57);a=torch.Generator().manual_seed(58);audit=torch.Generator().manual_seed(59)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=.001,weight_decay=0.)
    def step(token,target):
        optimizer.zero_grad(set_to_none=True)
        (h,_,packets,md),pending=actor.core.consume_for_outcome(actor.causes.embed(torch.tensor(token)),state,g,a)
        # Actual normalized native word/cause head; candidates are fixed from
        # the revealed prefix for this bounded task-credit contract.
        bank=NativeCauseBank(actor.depth,actor.width,32);bank.append(token,h,[],state['events']-1)
        candidates=bank.query(token,[],state['events']-1,maximum=8)
        def task(views,s,metadata):
            component,_,_=indexed_cause_nll(actor.causes,views,target,candidates,explore=False)
            prior=torch.log_softmax(actor.depth_query(views).squeeze(-1)+actor.depth_bias,-1)
            return -torch.logsumexp(prior-component,-1)
        nll=task(h,state,md)
        credit,rows=actor.core.outcome_credit(pending,nll,task,state,audit_generator=audit)
        objective=nll+credit;objective.backward();actor.deliver_credit(packets,state)
        norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        return float(nll.detach()),rows,float(norm)
    step(1,2);step(2,3)
    saved=io.BytesIO();torch.save((actor.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state(),audit.get_state(),torch.get_rng_state()),saved)
    expected=step(3,4);outcome=copy.deepcopy((actor.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state(),audit.get_state(),torch.get_rng_state()))
    saved.seek(0);weights,moments,state,gs,als,aus,rng=torch.load(saved,weights_only=False)
    actor.load_state_dict(weights);optimizer.load_state_dict(moments);g.set_state(gs);a.set_state(als);audit.set_state(aus);torch.set_rng_state(rng)
    actual=step(3,4);assert expected==actual
    assert identical(outcome,(actor.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state(),audit.get_state(),torch.get_rng_state()))
    assert state['direct_producer_replays']>0 and state['nonlinear_route_audits']>0


def test_reviewed_stream_keeps_episodic_writer_credit_and_recovers_all_learning():
    from experiments.native_query_migration import migrate
    from sleeping_machines.producer_replay_native_race import enable_producer_credit
    from sleeping_machines.document_boundary_causes import enable_document_relevance
    from sleeping_machines.coupled_native_query_credit import enable_coupled_query_credit
    from sleeping_machines.compositional_producer_credit import enable_compositional_credit
    from sleeping_machines.sampled_compositional_credit import enable_sampled_compositional_credit
    from sleeping_machines.deferred_episodic_writer_credit import enable_deferred_episodic_credit
    from sleeping_machines.integrated_review_learning import enable_reviewed_learning,ReviewedProducerCauseBank
    torch.manual_seed(17303)
    actor=LocalNativeDeepLM(8,torch.tensor([0,0,1,1,2,2,3,3]),torch.arange(8).reshape(4,2),width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,explore=1,dk=4,packed=True,bank_capacity=32)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=.001,weight_decay=0.)
    initial=actor.new_state();initial['cause_bank']=NativeCauseBank(actor.depth,actor.width,32);initial['cause_inbox']=torch.zeros(actor.depth,actor.width)
    actor,optimizer,state,_=migrate(actor,optimizer,initial)
    enable_producer_credit(actor,state);enable_document_relevance(actor,optimizer);enable_coupled_query_credit(actor,state)
    enable_compositional_credit(actor,state);enable_sampled_compositional_credit(actor,state);enable_deferred_episodic_credit(actor,state)
    ids=[id(p) for p in actor.parameters()];enable_reviewed_learning(actor,state);assert ids==[id(p) for p in actor.parameters()]
    g=torch.Generator().manual_seed(61);a=torch.Generator().manual_seed(62);tape=torch.tensor([1,2,1,3,1,2,4,1,3,2,1,4,2])
    def step(offset):
        optimizer.zero_grad(set_to_none=True)
        nll,objective,_,packets,_=actor.stream_nll(tape[offset:offset+2],tape[offset+1:offset+3],state,g,a)
        objective.backward();actor.deliver_credit(packets,state);norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        return nll.detach(),float(norm)
    step(0);step(2);step(4)
    assert isinstance(state['cause_bank'],ReviewedProducerCauseBank)
    assert state['episodic_writer_replays']>0 and state['direct_producer_replays']>0 and state['nonlinear_route_audits']>0
    saved=io.BytesIO();torch.save((actor.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state(),torch.get_rng_state()),saved)
    expected=step(6);outcome=copy.deepcopy((actor.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state(),torch.get_rng_state()))
    saved.seek(0);weights,moments,state,gs,als,rng=torch.load(saved,weights_only=False)
    actor.load_state_dict(weights);optimizer.load_state_dict(moments);g.set_state(gs);a.set_state(als);torch.set_rng_state(rng)
    actual=step(6);assert torch.equal(expected[0],actual[0]) and expected[1]==actual[1]
    assert identical(outcome,(actor.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state(),torch.get_rng_state()))
