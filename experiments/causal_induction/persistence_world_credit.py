"""Matched cheap belief intervention; native world and memory stay trained."""
from world_credit_sensor import WorldCreditSensor

class PersistenceWorldCreditSensor(WorldCreditSensor):
    def __init__(self, confidence, mode='reciprocal'):
        super().__init__(confidence, mode)
        self.persistence_total = 2050.
        self.persistence_correct = round(confidence*self.persistence_total)
        assert abs(self.persistence_correct/self.persistence_total-confidence) < 1e-12

    def fresh(self):
        st = super().fresh()
        st['persistence_correct'] = self.persistence_correct
        st['persistence_total'] = self.persistence_total
        st['persistence_receipts'] = []
        return st

    def issue(self, r, st, generator, arm='learned', learning=False):
        assert arm == 'learned'
        result = super().issue(r, st, generator, arm, learning)
        packet = st['issued'][r['query_id']]
        bit = st['bits'].get(r['node'], 0)
        confidence = max(.001, min(.999, st['persistence_correct']/st['persistence_total']))
        packet['persistence_bit'] = bit
        packet['persistence_probability'] = confidence if bit else 1-confidence
        return result

    def feedback(self, r, st, learning):
        packet = st['issued'][r['query_id']]
        native_q = packet['world_probability']
        cheap_q = packet['persistence_probability']
        # Only the stopped belief used in source-choice teaching is changed.
        # Native outcome logits, actual-outcome loss, and memory operations stay intact.
        packet['world_probability'] = cheap_q
        try:
            result = super().feedback(r, st, learning)
        finally:
            packet['world_probability'] = native_q
        y = float(r['target'])
        st['counts']['world_brier_sum'] += (native_q-y)**2-(cheap_q-y)**2
        st['persistence_receipts'].append(dict(query_id=r['query_id'],
            issued_time=packet['time'], probability=cheap_q,
            correct_before=st['persistence_correct'], total_before=st['persistence_total']))
        st['persistence_correct'] += int(packet['persistence_bit'] == r['target'])
        st['persistence_total'] += 1
        self.last_persistence_receipts = st['persistence_receipts']
        return result
