"""Native outcome memory keyed by target and an issue-time public source.
The source is an observed exogenous event, not a claimed hidden causal edge.
"""
from world_credit_sensor import WorldCreditSensor

class SourceAddressedWorldCredit(WorldCreditSensor):
    def fresh(self):
        st=super().fresh()
        st['last_public_source']=None
        st['counts'].update(source_addressed_world_events=0)
        return st

    def observe(self,r,st):
        super().observe(r,st)
        if r['exogenous']:
            st['last_public_source']=(r['node'],r['time'])

    def world_event(self,node,t,features,kind,outcome,st):
        source=st['active_world_source']
        st['counts']['source_addressed_world_events']+=1
        return self.credit_event((node,source),t,features,kind,0.,outcome,st)

    def issue(self,r,st,generator,arm='exact',learning=False):
        public=st['last_public_source']
        assert public is None or public[1]<=r['time']
        source=r['node'] if public is None else public[0]
        st['active_world_source']=source
        try:
            probability=super().issue(r,st,generator,arm,learning)
            st['issued'][r['query_id']]['world_source']=source
            return probability
        finally:
            st.pop('active_world_source')

    def feedback(self,r,st,learning=False):
        # Restore the issue-time binding, even if newer source events arrived.
        st['active_world_source']=st['issued'][r['query_id']]['world_source']
        try:
            return super().feedback(r,st,learning)
        finally:
            st.pop('active_world_source')
