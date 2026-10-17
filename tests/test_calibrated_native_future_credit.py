import copy
import io
import torch
from sleeping_machines.calibrated_native_future_credit import CalibratedNativeFutureCredit
from tests.test_native_query_migration import identical


def setup():
    torch.manual_seed(20171)
    model=CalibratedNativeFutureCredit(8,depth=3,heads=2,pool=3)
    return model,model.new_state(),torch.Generator().manual_seed(81),torch.Generator().manual_seed(82)


def test_neutral_service_independent_supported_audits_and_zero_action():
    model,state,g,a=setup();model.begin_decision()
    quote,_,_=model.quote(*(torch.randn(8) for _ in range(3)),.5,.001,state,g,a)
    assert torch.equal(quote,torch.zeros_like(quote))
    choice,uncertainty=model.choose_service(quote)
    assert choice==1 and uncertainty==0
    generator=torch.Generator().manual_seed(83)
    seen=set()
    for _ in range(32):
        j,p=model.choose_audit(generator);seen.add(j)
        assert j in [0,2] and p==.5
    assert seen=={0,2}


def test_recursive_response_matches_batch_ridge_and_complete_resume():
    model,state,g,a=setup()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=0.)
    vectors=[torch.randn(8) for _ in range(3)]
    rows=[];targets=[]
    for k in range(12):
        model.begin_decision()
        quote,_,packets=model.quote(*vectors,.5,.001,state,g,a)
        j=0 if k%2 else 2
        feature=torch.cat(model.features)[j].detach().double()
        target=feature@torch.arange(len(feature),dtype=torch.float64)/10
        rows.append(feature);targets.append(target)
        model.learn_audit(quote,j,target,packets,state,optimizer)
    X=torch.stack(rows);Y=torch.stack(targets)
    expected=torch.linalg.solve(X.T@X+.1*torch.eye(X.shape[1],dtype=torch.float64),X.T@Y)
    torch.testing.assert_close(model.response,expected,rtol=1e-10,atol=1e-10)
    assert torch.linalg.eigvalsh(model.covariance).min()>0
    buffer=io.BytesIO();torch.save((model.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state()),buffer)
    def update():
        model.begin_decision();quote,_,packets=model.quote(*vectors,.5,.001,state,g,a)
        return quote.detach(),model.learn_audit(quote,0,-.002,packets,state,optimizer)
    result=update();final=copy.deepcopy((model.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state()))
    buffer.seek(0);weights,moments,state,gs,als=torch.load(buffer,weights_only=False)
    model.load_state_dict(weights);optimizer.load_state_dict(moments);g.set_state(gs);a.set_state(als)
    actual=update()
    assert torch.equal(result[0],actual[0]) and result[1]==actual[1]
    assert identical(final,(model.state_dict(),optimizer.state_dict(),state,g.get_state(),a.get_state()))
