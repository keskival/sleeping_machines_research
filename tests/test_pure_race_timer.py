from types import SimpleNamespace
import torch
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel
from sleeping_machines.pure_race_timer import PureRaceTimer


def test_pure_timer_preserves_native_forward_race_bitwise():
    scores=torch.tensor([.2,-.5,.7],dtype=torch.double)
    for seed in range(16):
        torch.manual_seed(seed);_,old,winner=ParallelHeadRaceLanguageModel.race(scores)
        torch.manual_seed(seed);new,new_winner=PureRaceTimer.apply(scores)
        assert torch.equal(old,new) and torch.equal(winner,new_winner)


def test_new_timing_credit_is_conditional_mean_of_native_winner_credit():
    rates=torch.tensor([.3,1.,2.],dtype=torch.double);time=torch.tensor(.7,dtype=torch.double)
    error=torch.tensor(1.3,dtype=torch.double);pi=rates/rates.sum()
    ctx=SimpleNamespace(saved_tensors=(rates,time),input_dtype=torch.double)
    improved=PureRaceTimer.backward(ctx,error,None)
    common=-error*.010*time/(1+time).square()
    original=common*torch.eye(3,dtype=torch.double)
    mean=(pi[:,None]*original).sum(0)
    assert torch.allclose(improved,mean,atol=1e-15,rtol=1e-15)
    conditional_variance=(pi[:,None]*(original-mean).square()).sum()
    assert conditional_variance>0


def test_pure_timer_clock_loss_trains_all_candidate_rates():
    scores=torch.tensor([.2,-.5,.7],dtype=torch.double,requires_grad=True)
    delay,_=PureRaceTimer.apply(scores);delay.square().backward()
    assert torch.isfinite(scores.grad).all() and (scores.grad<0).all()
