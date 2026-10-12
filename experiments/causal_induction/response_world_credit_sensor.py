"""Native residual outcome law conditioned on already issued responses.

A separate development member: no parent sources/checkpoints are changed.
"""
import torch
from torch import nn
from world_credit_sensor import WorldCreditSensor

class ResponseWorldCreditSensor(WorldCreditSensor):
    def __init__(self, confidence, mode='reciprocal', response_enabled=True):
        super().__init__(confidence, mode)
        self.response_enabled = response_enabled
        with torch.random.fork_rng():
            self.critic.response_content = nn.Linear(3, self.core.total_payload, bias=False)
        nn.init.zeros_(self.response_content.weight)
        # The outcome model starts at the issued forward reference, then learns
        # corrections from actual outcomes. This is a new initialization, not
        # unchanged transport of an acquired parent outcome readout.
        nn.init.zeros_(self.critic.head.weight)
        nn.init.zeros_(self.critic.head.bias)

    @property
    def response_content(self):
        return self.critic.response_content

    def fresh(self):
        st = super().fresh()
        st['response_issued'] = {}
        st['counts'].update(response_inputs=0, response_feedback=0,
                           response_projection_madds=0, response_peak_issued_tensor_bytes=0,
                           response_projection_parameters=self.response_content.weight.numel())
        return st

    def issue(self, r, st, generator, arm='exact', learning=False):
        assert '_response_query' not in st
        st['_response_query'] = r['query_id']
        st['_response_decodes'] = []
        def capture(module, inputs, output):
            st['_response_decodes'].append(output.detach().reshape(()).clone())
        handle = self.sensor.register_forward_hook(capture)
        try:
            return super().issue(r, st, generator, arm, learning)
        finally:
            handle.remove()
            st.pop('_response_query')
            st.pop('_response_decodes')

    def feedback(self, r, st, learning):
        assert '_response_query' not in st
        st['_response_query'] = r['query_id']
        try:
            return super().feedback(r, st, learning)
        finally:
            st.pop('_response_query')
            st['response_issued'].pop(r['query_id'])

    def world_event(self, node, t, features, kind, outcome, st):
        query_id = st['_response_query']
        if kind == 0.:
            logits = st['_response_decodes']
            assert len(logits) in (1,2)
            factual = logits[0]
            alternative = logits[1] if len(logits)==2 else factual
            token = torch.stack((factual, alternative, alternative-factual)).detach().clone()
            st['response_issued'][query_id] = token
            st['counts']['response_peak_issued_tensor_bytes'] = max(st['counts']['response_peak_issued_tensor_bytes'], sum(x.numel()*x.element_size() for x in st['response_issued'].values()))
            st['counts']['response_inputs'] += 1
        else:
            token = st['response_issued'][query_id]
            st['counts']['response_feedback'] += 1
        reference = token[0].detach()
        if not self.response_enabled:
            token = torch.zeros_like(token)
        extra = .1*self.response_content(token).tanh()
        p = self.core.total_payload
        # Inject after the parent's observational detach boundary so actual
        # outcome teaching reaches the adapter and native read policy.
        def condition_content(module, inputs):
            content = inputs[0]
            return (torch.cat((content[:p]+extra, content[p:])),)
        handle = self.critic.content.register_forward_pre_hook(condition_content)
        try:
            residual = super().world_event(node,t,features,kind,outcome,st)
        finally:
            handle.remove()
        st['counts']['response_projection_madds'] += self.response_content.weight.numel()
        return reference+residual
