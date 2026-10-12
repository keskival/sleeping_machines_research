"""One native causal world belief, coherent credit projections for all routes.

Known BCE algebra projects the learned outcome law. This first integration
changes source-choice teaching, not internal exact/surrogate actor gradients.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from addressed_shared_bank import AddressedSharedBank
from sleeping_machines.paired_route_credit import paired_route_credit


class WorldCreditSensor(AddressedSharedBank):
    def __init__(self, confidence, mode='reciprocal'):
        super().__init__(confidence, mode)
        # Outcome supervision trains the native body as well as the readout.
        old = self.critic.head
        head = nn.Linear(old.weight.shape[1], old.weight.shape[0])
        with torch.no_grad():
            head.weight.copy_(old.weight)
            head.bias.copy_(old.bias)
        self.critic.head = head

    def fresh(self):
        st = super().fresh()
        st['counts'].update(world_queries=0, world_feedback=0,
                           world_nll_sum=0., world_brier_sum=0.,
                           route_logit_gap_squared=0.,
                           route_contrast_residual_squared=0.,
                           route_label_noise_proxy=0.)
        return st

    def world_event(self, node, t, features, kind, outcome, st):
        # One target-addressed state, not one independent loss model per route.
        # The third field carries actual outcomes only at feedback delivery.
        return super().credit_event((node, node), t, features, kind, 0., outcome, st)

    def issue(self, r, st, generator, arm='exact', learning=False):
        assert arm in ('exact', 'learned', 'prior')
        assert 'target' not in r and r['query_id'] not in st['issued']
        node, t = r['node'], r['time']
        target = self.context(node, t, st)
        p = self.core.total_payload
        zero = target.new_tensor([float(st['bits'].get(node, 0)),
                                 math.log1p(max(0., t-st['last'].get(node, t))),
                                 math.log1p(r['horizon'])])
        query = self.query(torch.cat((target, zero)))
        candidates = [node]+[n for n in st['recent'] if n != node][:self.candidates-1]
        contexts = [self.context(n, t, st) for n in candidates]
        aux = [target.new_tensor([float(st['bits'].get(n, 0)),
                                  math.log1p(max(0., t-st['last'].get(n, t)))])
               for n in candidates]
        edges = []
        for source in candidates:
            count, same, transition, visits = st['edges'].get((node, source), [0.]*4)
            edges.append(target.new_tensor([math.log1p(count), same/(count+1.),
                                            transition/(count+1.), count/(visits+1.)]))
        scores = torch.stack([(query*self.key(torch.cat((v, a)))).sum()/math.sqrt(p)
                              +self.edge_score(e).squeeze()
                              for v, a, e in zip(contexts, aux, edges)]).clamp(-12, 12)
        pi = scores.softmax(0)
        noise = torch.empty(len(candidates), dtype=torch.float64).exponential_(generator=generator)
        winner = int((noise/scores.double().exp()).argmin())
        features = [torch.cat((target, v, zero, a, e))
                    for v, a, e in zip(contexts, aux, edges)]
        factual = self.sensor(features[winner]).squeeze()
        alternative = alt = None
        if learning and len(candidates) > 1:
            tg = st.setdefault('teacher_generator', torch.Generator().manual_seed(812))
            choices = [j for j in range(len(candidates)) if j != winner]
            alternative = choices[int(torch.randint(len(choices), (1,), generator=tg))]
            with torch.no_grad():
                alt = self.sensor(features[alternative]).squeeze()
        # Pool only already-discovered public content. No chosen-route/label input.
        # This bounded summary is not a sufficient-statistic theorem.
        world_features = torch.cat((target.detach(), torch.stack(contexts).detach().mean(0),
                                    zero, torch.stack(aux).mean(0), torch.stack(edges).mean(0)))
        world_logit = self.world_event(node, t, world_features, 0., 0., st)
        c = st['counts']
        c['queries'] += 1
        c['candidate_keys'] += len(candidates)
        c['discovery_context_reads'] += len(candidates)+1
        c['relationship_reads'] += len(candidates)
        c['value_reads'] += 1+int(alt is not None)
        c['sensor_decodes'] += 1+int(alt is not None)
        c['exact_alternative_audits'] += int(alt is not None)
        c['world_queries'] += 1
        st['issued'][r['query_id']] = dict(
            time=t, node=node, logit=factual, probability=float(factual.detach().sigmoid()),
            pi=pi, winner=winner, alternative=alternative, alt=alt,
            proposal=1/max(1, len(candidates)-1),
            world_logit=world_logit, world_features=world_features.detach(),
            world_probability=float(world_logit.detach().sigmoid()), arm=arm)
        return st['issued'][r['query_id']]['probability']

    @staticmethod
    def contrast(z_alt, z_factual, mean_target):
        return F.softplus(z_alt)-F.softplus(z_factual)-mean_target*(z_alt-z_factual)

    def feedback(self, r, st, learning):
        packet = st['issued'].pop(r['query_id'])
        assert packet['time'] < r['time']
        y = packet['logit'].new_tensor(float(r['target']))
        actor_loss = F.binary_cross_entropy_with_logits(packet['logit'], y)
        world_loss = F.binary_cross_entropy_with_logits(packet['world_logit'], y)
        c = st['counts']
        c['world_nll_sum'] += float(world_loss.detach())
        c['world_brier_sum'] += (packet['world_probability']-float(y))**2
        if learning and packet['alternative'] is not None:
            factual = packet['logit'].detach()
            alternative = packet['alt'].detach()
            issued_mean = (.5 if packet['arm'] == 'prior'
                           else packet['world_probability'])
            mean_target = (y if packet['arm'] == 'exact'
                           else .5*y+.5*issued_mean)
            delta = self.contrast(alternative, factual, mean_target)
            actual = self.contrast(alternative, factual, y)
            c['route_logit_gap_squared'] += float((alternative-factual).square())
            c['route_contrast_residual_squared'] += float((delta-actual).square())
            c['route_label_noise_proxy'] += (packet['world_probability']
                *(1-packet['world_probability'])*float((alternative-factual).square()))
            actor_loss = actor_loss+paired_route_credit(
                packet['pi'][None], torch.tensor([packet['alternative']]),
                y.new_tensor([packet['proposal']]), delta[None])
        # The delivered outcome updates exactly one native target memory.
        self.world_event(packet['node'], r['time'], packet['world_features'],
                         1., float(y), st)
        c['world_feedback'] += 1
        c['critic_targets'] += int(learning)
        return actor_loss+(world_loss if learning else 0.), packet['probability']
