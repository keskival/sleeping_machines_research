"""Typed predicate events in the existing integrated temporal/sparse core.

Stage-one integration: fixed predicates, canonical full predicate presentation,
learned receiver selection and actual alternative-write continuation credit.
No learned predicate discovery/thresholds or sparse predicate-reading claim.
"""
import torch
from torch import nn
from .typed_predicate_interface import TypedPredicateBank
from .addressed_event_heads import AddressedEventHeads
from .fast_native_core import fast_class
from .packed_token_core import PackedTokenCore
from .sparse_counterfactual_episodes import token_features
from .paired_route_credit import paired_route_credit
from .event_credit_sites import sample_site


class TypedTemporalModel(nn.Module):
    outcomes = ('false', 'true', 'missing', 'unknown')

    def __init__(self, predicates, classes=2, payload=8, depth=2, heads=2, pool=4):
        super().__init__()
        predicates = sorted(predicates, key=lambda p: p.name)
        if not predicates or classes < 2 or pool < 2:
            raise ValueError('Nonempty predicate bank, classification and alternatives required')
        self.bank = TypedPredicateBank(predicates)
        self.predicates = predicates
        self.query_id = 4 * len(predicates)
        core = fast_class(AddressedEventHeads)(sources=1, content_dim=self.query_id+1,
            classes=1, payload=payload, depth=depth, heads=heads, pool=pool)
        core.head = nn.Identity()
        self.core = PackedTokenCore(core)
        self.readout = nn.Linear(self.core.total_payload, classes)

    def encode(self, rows):
        """Select comparison outcomes by computational race, never raw mixing.

        All fixed predicates are read; sparse core writes are a separate count.
        Canonical order is a static-row computation policy, not physical time.
        """
        encoded = []
        for row in rows:
            ids = []
            for i, pred in enumerate(self.predicates):
                comparison = pred.compare(row)
                delays = comparison.race_delays()
                # Dict order implements the documented false-on-equality policy.
                winner = min(delays, key=delays.get)
                if winner != comparison.outcome:
                    raise ArithmeticError('Logical and computational race outcomes differ')
                ids.append(4*i + self.outcomes.index(winner))
            encoded.append(ids + [self.query_id])
        if not encoded:
            raise ValueError('A nonempty independent-row batch is required')
        return torch.tensor(encoded, dtype=torch.long)

    def forward(self, ids, generator, *, deterministic=False, force_site=None):
        if ids.ndim != 2 or ids.shape[1] != len(self.predicates)+1:
            raise ValueError('Canonical predicate events plus one terminal query required')
        features, state, probabilities = token_features(self.core, ids, state=None,
            generator=generator, route_credit='none', deterministic=deterministic,
            force_site=force_site)
        return self.readout(features[:, -1]), state, probabilities

    def fitting_loss(self, ids, targets, generator, site_generator, alternative_generator):
        """One sampled actual forced-write replay; all keys and fitting paid.

        No raw targets enter encoding or state. Common future race noise matches
        factual/alternative continuations. Route score credit uses terminal risk;
        alternative-value gradients are not supplied by this estimator.
        """
        before = generator.get_state()
        logits, state, probabilities = self(ids, generator)
        factual = nn.functional.cross_entropy(logits, targets, reduction='none')
        event, depth, head, inverse = sample_site(ids.shape[1], self.core.depth,
                                                 self.core.heads, site_generator)
        pi = probabilities[event*self.core.depth+depth][1][:, head].double()
        winner = state['race_winners'][event, depth, :, head]
        eligible = 1 - nn.functional.one_hot(winner, self.core.pool).double()
        conditional = pi.detach()*eligible
        mass = conditional.sum(-1, keepdim=True)
        conditional = torch.where(mass > 0, conditional/mass.clamp_min(1e-300),
                                  eligible/(self.core.pool-1))
        proposal = .9*conditional + .1*eligible/(self.core.pool-1)
        alternative = torch.multinomial(proposal, 1, generator=alternative_generator).squeeze(1)
        q = proposal.gather(1, alternative[:, None]).squeeze(1)
        shadow_generator = torch.Generator().set_state(before)
        with torch.no_grad():
            shadow, _, _ = self(ids, shadow_generator,
                                force_site=(event, depth, head, alternative))
            alternative_loss = nn.functional.cross_entropy(shadow, targets, reduction='none')
        difference = alternative_loss.double()-factual.detach().double()
        credit = inverse*paired_route_credit(pi, alternative, q, difference)
        return factual.mean()+credit, dict(site=[event, depth, head],
            inverse_site_probability=inverse, winner=winner.tolist(),
            alternative=alternative.tolist(), proposal_probability=q.tolist(),
            terminal_loss_difference=difference.tolist(),
            factual_nll=float(factual.mean().detach()),
            selected_updates=int(state['last_writes'].sum()),
            comparisons_per_row=len(self.predicates),
            available_predicates=len(self.predicates),
            scored_keys_per_event=self.core.depth*self.core.heads*self.core.pool,
            replay_events=ids.numel(),
            scope='Fixed predicate bank; full comparisons, sparse learned state updates; one actual alternative-write continuation teacher. No FLOP or tabular win claim.')
