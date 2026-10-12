"""Native outcome model conditioned on causal activation-distribution history.

Separate development member; does not alter the parent's admitted recipe.
History statistics are detached numerical inputs, not a replacement critic.
"""
import math
import numpy as np
import torch
from torch import nn

from world_credit_sensor import WorldCreditSensor
from sleeping_machines.route_activation_memory import RouteActivationMemory


class HistoryWorldCreditSensor(WorldCreditSensor):
    def __init__(self, confidence, mode='reciprocal', history_enabled=True):
        super().__init__(confidence, mode)
        self.history_enabled = history_enabled
        p = self.core.total_payload
        self.history_width = 2*p
        self.history_rank = min(4, self.history_width)
        self.history_dim = 2*self.history_width+self.history_rank*(self.history_rank+1)//2+4
        # Adding a zero adapter neither changes actor RNG nor initial forecasts.
        with torch.random.fork_rng():
            self.history_content = nn.Linear(self.history_dim, p, bias=False)
        nn.init.zeros_(self.history_content.weight)

    def fresh(self):
        st = super().fresh()
        projection = np.eye(self.history_width)[:self.history_rank]
        st['activation_history'] = RouteActivationMemory(self.history_width, projection, max_routes=1024)
        st['issued_history'] = {}
        st['counts'].update(history_queries=0, history_feedback=0, history_operator_version_reads=0)
        return st

    def _history_version(self, st):
        # Monotone in-process edit counters, not a portable learner fingerprint.
        versions = [p._version for p in self.parameters()]
        st['counts']['history_operator_version_reads'] += len(versions)
        return str(sum(versions))

    def issue(self, r, st, generator, arm='exact', learning=False):
        assert '_history_query' not in st
        st['_history_query'] = r['query_id']
        try:
            return super().issue(r, st, generator, arm, learning)
        finally:
            st.pop('_history_query')

    def feedback(self, r, st, learning):
        assert '_history_query' not in st
        st['_history_query'] = r['query_id']
        try:
            return super().feedback(r, st, learning)
        finally:
            st.pop('_history_query')
            st['issued_history'].pop(r['query_id'], None)

    def world_event(self, node, t, features, kind, outcome, st):
        query_id = st['_history_query']
        if kind == 0.:
            memory = st['activation_history']
            version = self._history_version(st)
            snapshot = memory.snapshot(node, t, version)
            token = features.new_zeros(self.history_dim)
            if snapshot is not None:
                upper = np.triu_indices(self.history_rank)
                values = np.r_[snapshot.mean, np.log1p(snapshot.variance),
                               snapshot.projected_covariance[upper],
                               math.log1p(snapshot.effective_count), math.log1p(snapshot.age),
                               1., float(snapshot.operator_stale)]
                token = features.new_tensor(values)
            st['issued_history'][query_id] = token.detach().clone()
            # Read-before-write: current activation cannot masquerade as history.
            memory.observe(node, features[:self.history_width].detach().cpu().numpy(), t, version)
            st['counts']['history_queries'] += 1
        else:
            token = st['issued_history'][query_id]
            st['counts']['history_feedback'] += 1
        if not self.history_enabled:
            token = torch.zeros_like(token)
        p = self.core.total_payload
        extra = .1*self.history_content(token).tanh()
        # Parent intentionally detaches observational tokens. Inject only after
        # that boundary, preserving actor isolation and the native read policy.
        def condition_content(module, inputs):
            content = inputs[0]
            return (torch.cat((content[:p]+extra, content[p:])),)
        handle = self.critic.content.register_forward_pre_hook(condition_content)
        try:
            return super().world_event(node, t, features, kind, outcome, st)
        finally:
            handle.remove()
