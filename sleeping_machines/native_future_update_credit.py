"""Native reciprocal update critic trained by controlled future consequences.

The forward actor supplies state/action context and revealed continuation
advantages. This persistent race machine ranks addressed forward-weight steps.
It is an action-value learner, not an exact full-history gradient oracle.
"""
import math
import torch
from torch import nn
from .queryable_local_deep_race import QueryableLocalDeepRace
from .sampled_compositional_credit import SampledCompositionalCredit


class NativeFutureUpdateCredit(nn.Module):
    def __init__(self, width, depth=8, heads=4, pool=4):
        super().__init__()
        self.width = width
        self.context = nn.Linear(3*width+3, width)
        self.core = SampledCompositionalCredit(QueryableLocalDeepRace(width, depth, heads, pool))
        self.action = nn.Linear(2, width, bias=False)
        self.value = nn.Linear(width, 1)

    def new_state(self):
        return self.core.new_state()

    def quote(self, published, producer_input, gradient_summary, address, step_rms,
              state, generator, alternative_generator, *, timestamp=None):
        if any(x.shape != (self.width,) for x in [published, producer_input, gradient_summary]):
            raise ValueError('state-interface vectors required, not coordinate averages')
        stamp=float(state['events']) if timestamp is None else float(timestamp)
        clocks=state.setdefault('interface_observation_times',{})
        elapsed=stamp-clocks.get(address,stamp)
        if not math.isfinite(stamp) or elapsed<0:
            raise ValueError('causal interface observation time required')
        clocks[address]=stamp
        context = torch.cat([published.detach(), producer_input.detach(), gradient_summary.detach(),
                             published.new_tensor([address, math.log1p(step_rms), math.log1p(elapsed)])])
        views, state, packets, _ = self.core.consume(self.context(context), state, generator,
                                                    alternative_generator, learn=True)
        # Direct readout from every native credit depth; no recurrent BPTT.
        representation = views.mean(0)
        controls = published.new_tensor([[-1., step_rms], [0., 0.], [1., step_rms]])
        raw = self.value(torch.tanh(representation[None]+self.action(controls))).squeeze(-1)
        return raw-raw[1], state, packets

    @staticmethod
    def choose(quotes, generator, *, temperature=.01, exploration=.2):
        if quotes.ndim != 1 or not len(quotes) or not bool(torch.isfinite(quotes).all()):
            raise ValueError('finite action costs required')
        if not 0 < exploration <= 1 or temperature <= 0:
            raise ValueError('positive action exploration required')
        probabilities = (1-exploration)*torch.softmax(-quotes.detach().double()/temperature, 0)
        probabilities = probabilities+exploration/len(quotes)
        selected = int(torch.multinomial(probabilities, 1, generator=generator))
        return selected, float(probabilities[selected])

    def learn_revealed(self, prediction, future_loss_difference, packets, state, optimizer):
        target = torch.as_tensor(future_loss_difference, dtype=prediction.dtype, device=prediction.device).detach()
        if target.ndim or not bool(torch.isfinite(target)):
            raise ValueError('actual revealed scalar continuation advantage required')
        optimizer.zero_grad(set_to_none=True)
        loss = (prediction-target).square()
        loss.backward()
        self.core.deliver_credit(packets, state)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return dict(squared_error=float(loss.detach()), preclip_norm=float(norm),
                    target_advantage=float(target), predicted_advantage=float(prediction.detach()))


def addressed_mix_action(model, gradients, depth, *, step_rms=.001):
    """A finite addressed intervention, in addition to ordinary actor learning.

    Both signs and no-op are quoted. The initial integration investigates this
    action interface; it does not claim coverage of arbitrary updates/routes.
    """
    if not 0 <= depth < model.depth or not 0 < step_rms or not math.isfinite(step_rms):
        raise ValueError('valid depth and finite positive step size required')
    gradient = gradients['core.mix'][depth].detach()
    scale = gradient.square().mean().sqrt()
    if float(scale) == 0:
        return torch.zeros_like(gradient)
    return step_rms*gradient/scale
