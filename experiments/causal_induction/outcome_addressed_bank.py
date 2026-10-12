"""Causal outcome-preserving credit tokens, with no added payload dimension."""
from addressed_shared_bank import AddressedSharedBank


class OutcomeAddressedBank(AddressedSharedBank):
 def credit_event(self,pair,t,features,kind,loss,audited,st):
  # Old audit flag duplicated kind: every delivered event had(kind,audit)=(1,1).
  # Replace it with the actual revealed target; request tokens contain none.
  information=st['revealed_outcome'] if kind==1. else 0.
  return super().credit_event(pair,t,features,kind,loss,information,st)

 def feedback(self,r,st,learning):
  assert 'revealed_outcome' not in st
  st['revealed_outcome']=float(r['target'])
  try:return super().feedback(r,st,learning)
  finally:st.pop('revealed_outcome')
