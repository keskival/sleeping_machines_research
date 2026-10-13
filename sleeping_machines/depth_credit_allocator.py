"""Predictable full-support allocation of native language route-credit audits.

Only audit sites change. Forward races and factual gradients are unchanged.
The allocation statistic is a local score-gradient proxy, not whole-parameter
gradient variance. Positive exploration and exact site weights retain the
conditional raw audit mean before global clipping/Adam.
"""
import math
import torch


class DepthCreditAllocator:
    def __init__(self, depth, heads, window, exploration=.25, decay=.95):
        if depth < 8 or heads < 1 or window < 1:
            raise ValueError('This R1 candidate targets actual temporal depth >= 8')
        if not 0 < exploration <= 1 or not 0 <= decay < 1:
            raise ValueError('Positive support and a valid EMA are required')
        self.depth, self.heads, self.window = depth, heads, window
        self.exploration, self.decay = exploration, decay
        self.second = torch.ones(depth, dtype=torch.float64)
        self.counts = torch.zeros(depth, dtype=torch.long)
        self.last = None
        self.rows = []

    def probabilities(self):
        # This distribution is computed before the current outcome is observed.
        importance = self.second.clamp_min(1e-30).sqrt()
        return (1-self.exploration)*importance/importance.sum() + self.exploration/self.depth

    def sample_site(self, length, depth, heads, generator):
        if (depth, heads) != (self.depth, self.heads) or length < 1:
            raise ValueError('Allocator dimensions changed')
        probabilities = self.probabilities()
        layer = int(torch.multinomial(probabilities, 1, generator=generator))
        token = int(torch.randint(length, (), generator=generator))
        head = int(torch.randint(heads, (), generator=generator))
        probability = float(probabilities[layer]) / (length*heads)
        self.last = dict(token=token, layer=layer, head=head, length=length,
                         probability=probability, layer_probability=float(probabilities[layer]))
        return token, layer, head, 1/probability

    @torch.no_grad()
    def observe(self, probabilities, alternative, proposal_probability, difference):
        if self.last is None:
            raise RuntimeError('Outcome arrived without an issued audit site')
        item = self.last
        token, length = item['token'], item['length']
        end = min(length, (token//self.window+1)*self.window)
        suffix_weight = (end-token)/length
        pi = probabilities.detach().double()
        pj = pi.gather(1, alternative[:, None]).squeeze(1)
        score = torch.nn.functional.one_hot(alternative, pi.shape[1]).double()-pi
        coefficient = suffix_weight*difference.detach().double()/proposal_probability.detach().double()
        proxy = (coefficient[:, None]*pj[:, None]*score/pi.shape[0]).square().sum()
        value = float(proxy)
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite audit allocation statistic')
        layer = item['layer']
        # Initialize each layer from its first observed audit, then use its own
        # observed conditional moment. Do not train allocation on DEV feedback.
        self.second[layer] = value if self.counts[layer] == 0 else self.decay*self.second[layer]+(1-self.decay)*value
        self.counts[layer] += 1
        self.rows.append(dict(**item, score_gradient_proxy_square=value))
        self.last = None

    def packet(self):
        return dict(exploration=self.exploration, decay=self.decay,
                    layer_audits=self.counts.tolist(), final_layer_probabilities=self.probabilities().tolist(),
                    second_moment_proxy=self.second.tolist(), issued_sites=self.rows,
                    scope='TRAIN-only predictable layer allocation; tokens/heads uniform. '
                          'Local probability-score proxy, not full-parameter variance or an efficiency result.')

    def state_dict(self):
        if self.last is not None:
            raise RuntimeError('Checkpoint only after issued feedback has matured')
        return dict(second=self.second.clone(), counts=self.counts.clone(), rows=list(self.rows),
                    configuration=(self.depth,self.heads,self.window,self.exploration,self.decay))

    def load_state_dict(self, saved):
        if tuple(saved['configuration']) != (self.depth,self.heads,self.window,self.exploration,self.decay):
            raise ValueError('Exact resume requires unchanged allocation settings')
        self.second=saved['second'].clone()
        self.counts=saved['counts'].clone()
        self.rows=list(saved['rows'])
        self.last=None


def check_weight_contract():
    """Exact finite sum of nonuniform site estimates at depths8/16."""
    for depth in (8, 16):
        allocator = DepthCreditAllocator(depth, 2, 16)
        allocator.second = torch.arange(1, depth+1, dtype=torch.float64).square()
        p = allocator.probabilities()
        assert bool((p >= allocator.exploration/depth).all())
        assert abs(float(p.sum())-1) < 1e-14
        length = 5
        demand = torch.arange(length*depth*2, dtype=torch.float64).reshape(length, depth, 2)-7
        expected = sum(float(p[d]/(length*2)) * float(demand[t,d,h]) / float(p[d]/(length*2))
                       for t in range(length) for d in range(depth) for h in range(2))
        assert abs(expected-float(demand.sum())) < 1e-9
        # Every proposed layer receives finite positive support at extreme moments.
        allocator.second[0] = 1e30
        assert bool((allocator.probabilities() >= allocator.exploration/depth).all())
        generator=torch.Generator().manual_seed(37)
        allocator.sample_site(5,depth,2,generator)
        logits=torch.tensor([[.2,.3,-.1],[.6,-.4,.2]],dtype=torch.float64,requires_grad=True)
        pi=logits.softmax(-1)
        alternative=torch.tensor([1,2])
        proposal=torch.tensor([.4,.7],dtype=torch.float64)
        difference=torch.tensor([.03,-.07],dtype=torch.float64)
        token=allocator.last['token']
        end=min(5,(token//allocator.window+1)*allocator.window)
        loss=((end-token)/5)*(pi.gather(1,alternative[:,None]).squeeze(1)*difference/proposal).mean()
        gradient=torch.autograd.grad(loss,logits)[0]
        allocator.observe(pi,alternative,proposal,difference)
        assert abs(allocator.rows[-1]['score_gradient_proxy_square']-float(gradient.square().sum()))<1e-15
        saved=allocator.state_dict();tape=generator.get_state()
        issued=allocator.sample_site(5,depth,2,generator)
        restored=DepthCreditAllocator(depth,2,16)
        restored.load_state_dict(saved)
        other=torch.Generator();other.set_state(tape)
        assert restored.sample_site(5,depth,2,other)==issued
