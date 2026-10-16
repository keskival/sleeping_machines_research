"""Task-cotangent-directed credit to live version-bound episodic producers.

One conditional producer read per event, after scoring the outcome. Numerical
bank capacity/history is unchanged; missing producer records are not invented.
Version metadata is retained only while the existing binding store retains it.
"""
import copy
from dataclasses import dataclass, field
import math
import sys
import types
import torch
from .native_cause_bank import NativeCauseBank
from .document_boundary_causes import BoundaryHeadView, document_cause_law
from .episodic_writer_cause_head import indexed_cause_nll_with_writer_features


@dataclass
class EpisodicCreditBatch:
    packets: list = field(default_factory=list)
    delivered: bool = False


class ProducerCauseBank(NativeCauseBank):
    def append(self, *args, **kwargs):
        serial = super().append(*args, **kwargs)
        ctx = getattr(self, '_writer_context', None)
        if ctx is not None:
            state = ctx['state']
            live = state['bindings'].entries
            self.writer_versions = {s: keys for s, keys in self.writer_versions.items()
                                    if s >= self.count-self.capacity and any(k in live for k in keys)}
            self.writer_versions[serial] = tuple(bank[-1] for bank in state['publications'])
        return serial

    def query(self, *args, **kwargs):
        result = super().query(*args, **kwargs)
        ctx = getattr(self, '_writer_context', None)
        if ctx is not None and ctx['learn']:
            state = ctx['state']
            eligible = [(i, d, key)
                        for i, serial in enumerate(result['serials'].tolist())
                        for d, key in enumerate(self.writer_versions.get(serial, ()))
                        if key in state['bindings'].entries and key in state['producer_records']]
            if eligible:
                # Entries and producer records are immutable numerical versions.
                # Retain references for this credit batch, not a stream graph.
                snapshot = {k: v for k, v in state.items() if isinstance(v, int)}
                snapshot['bindings'] = copy.copy(state['bindings'])
                snapshot['bindings'].entries = dict(state['bindings'].entries)
                for name in ['producer_records', 'compositional_lineage']:
                    if name in state:
                        snapshot[name] = dict(state[name])
                result['_writer_credit'] = (ctx, snapshot, eligible)
                state['episodic_writer_snapshot_references'] += len(snapshot['bindings'].entries)
        return result

    def storage(self):
        result = super().storage()
        result.update(writer_provenance_events=len(self.writer_versions),
                      writer_provenance_references=sum(len(v) for v in self.writer_versions.values()))
        result['scope'] += ' Producer-version metadata and temporary credit-batch snapshots are additional; numerical history is unchanged.'
        return result


def deferred_writer_law(head, views, target, candidates, **kwargs):
    if '_writer_credit' not in candidates:
        return document_cause_law(head, views, target, candidates, **kwargs)
    ctx, snapshot, eligible = candidates['_writer_credit']
    features = candidates['features'].detach().requires_grad_(True)
    candidate = dict(candidates, _writer_features=features)
    fired = False

    def credit(gradient):
        nonlocal fired
        if fired:
            raise RuntimeError('episodic writer credit supports one backward per scored event')
        fired = True
        core, state = ctx['core'], ctx['state']
        if any(p._version != version for p, version in ctx['versions']):
            raise RuntimeError('parameters changed between prediction and writer credit')
        # Selection occurs only after the issued forecast receives its outcome.
        # Cotangent norms are a proxy for parameter-credit norms, not an optimum.
        norms = torch.stack([gradient[i, d].detach().double().norm() for i, d, _ in eligible])
        if not bool(torch.isfinite(norms).all()):
            raise RuntimeError('nonfinite episodic feature cotangent')
        total = norms.sum()
        if float(total) == 0:
            state['episodic_writer_zero_cotangent_events'] += 1
            return gradient
        floor = ctx['floor']
        probabilities = (1-floor)*norms/total + floor/len(eligible)
        generator = torch.Generator()
        generator.set_state(state['episodic_writer_rng'])
        selected = int(torch.multinomial(probabilities, 1, generator=generator))
        state['episodic_writer_rng'] = generator.get_state()
        i, d, key = eligible[selected]
        probability = float(probabilities[selected])
        before = {k: v for k, v in snapshot.items() if isinstance(v, int)}
        reads_before = snapshot['bindings'].reads
        if 'composition_credit_rng' in state:
            snapshot['composition_credit_rng'] = state['composition_credit_rng'].clone()
        with torch.enable_grad():
            value = core.read_publication(snapshot, key, snapshot['bindings'].entries[key].available,
                                          ctx['batch'].packets, learn=True)
            torch.autograd.backward(value, gradient[i, d].detach().to(value.dtype)/probability)
        for k, v in snapshot.items():
            if isinstance(v, int):
                if '_max_' in k or k.endswith('_max'):
                    state[k] = max(state.get(k, 0), v)
                else:
                    state[k] = state.get(k, 0) + v-before.get(k, 0)
        state['bindings'].reads += snapshot['bindings'].reads-reads_before
        if 'composition_credit_rng' in snapshot:
            state['composition_credit_rng'] = snapshot['composition_credit_rng']
        state['episodic_writer_replays'] += 1
        state['episodic_writer_max_inverse_probability'] = max(
            state['episodic_writer_max_inverse_probability'], 1/probability)
        return gradient

    features.register_hook(credit)
    actual_head = BoundaryHeadView(head, candidates['same_document']) if hasattr(head, 'document_bias') else head
    return indexed_cause_nll_with_writer_features(actual_head, views, target, candidate, **kwargs)


def enable_deferred_episodic_credit(model, state, *, seed=19193, uniform_floor=.1):
    if not 0 < uniform_floor <= 1 or not math.isfinite(uniform_floor):
        raise ValueError('positive proposal support required')
    if hasattr(model, '_episodic_writer_enabled'):
        raise ValueError('episodic writer credit already enabled')
    if not hasattr(model.core, 'producer_inputs'):
        raise ValueError('versioned native producer replay required')
    identities = [id(p) for p in model.parameters()]
    state.setdefault('episodic_writer_rng', torch.Generator().manual_seed(seed).get_state())
    for key in ['episodic_writer_replays', 'episodic_writer_zero_cotangent_events',
                'episodic_writer_snapshot_references']:
        state.setdefault(key, 0)
    state.setdefault('episodic_writer_max_inverse_probability', 0.)
    original_stream, original_deliver = model.stream_nll, model.deliver_credit

    def stream(self, inputs, targets, numerical_state, generator, alternative_generator, *, learn=True, stats=False):
        bank = numerical_state.get('cause_bank')
        if bank is None:
            bank = NativeCauseBank(self.depth, self.width, self.bank_capacity)
            numerical_state['cause_bank'] = bank
            numerical_state['cause_inbox'] = self.causes.embed.weight.new_zeros(self.depth, self.width)
        if not isinstance(bank, ProducerCauseBank):
            if type(bank) is not NativeCauseBank:
                raise ValueError('unrecognized episodic bank schema')
            bank.__class__ = ProducerCauseBank
            bank.writer_versions = {}  # preserve old data; no invented provenance
        batch = EpisodicCreditBatch()
        bank._writer_context = dict(core=self.core, state=numerical_state, batch=batch,
                                    learn=learn and torch.is_grad_enabled(), floor=uniform_floor,
                                    versions=[(p, p._version) for p in self.parameters()])
        try:
            nll, obj, numerical_state, packets, report = original_stream(
                inputs, targets, numerical_state, generator, alternative_generator, learn=learn, stats=stats)
        finally:
            del bank._writer_context
        packets.append(batch)
        return nll, obj, numerical_state, packets, report

    def deliver(self, packets, numerical_state):
        expanded = []
        for packet in packets:
            if isinstance(packet, EpisodicCreditBatch):
                if packet.delivered:
                    raise RuntimeError('episodic credit batch already delivered')
                packet.delivered = True
                expanded.extend(packet.packets)
            else:
                expanded.append(packet)
        return original_deliver(expanded, numerical_state)

    model.stream_nll = types.MethodType(stream, model)
    model.deliver_credit = types.MethodType(deliver, model)
    model._episodic_writer_enabled = True
    import experiments.r1_local_native_deep as canonical
    canonical.indexed_cause_nll = deferred_writer_law
    if 'r1_local_native_deep' in sys.modules:
        sys.modules['r1_local_native_deep'].indexed_cause_nll = deferred_writer_law
    assert identities == [id(p) for p in model.parameters()]
    return dict(new_parameters=0, numerical_memory_reset=False,
                replay='One available episodic producer per event after cotangent arrival; positive-support norm-informed proposal and inverse-probability credit.',
                scope='Conditional content credit to available versioned producers only. No historical route/clock credit, missing provenance recovery, whole-history exactness or unbiased clipped/Adam update claim. Extra snapshots, replay and own-write credit are paid.')
