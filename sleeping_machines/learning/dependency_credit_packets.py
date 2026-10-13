"""Dependency-stamped local delivery; no implicit global parameter epoch.

Read sets must be complete for the declared credit target. Sparse metadata is
not proof that unlisted dependencies have no effect. Uncertainty/transport of
stale credit belongs to a separately trained and audited acceptance policy.
"""
from dataclasses import dataclass
from .local_credit_banks import CreditPacket


@dataclass(frozen=True)
class DependencyCreditPacket:
    credit: CreditPacket
    parameter_versions: tuple
    state_versions: tuple
    target_id: str


class DependencyBankDelivery:
    def __init__(self,banks):
        self.banks=banks;self.state_versions={};self.accepted=0;self.rejected=0
    def register_state(self,address):
        if address in self.state_versions:raise ValueError('State address already registered')
        self.state_versions[address]=0
    def state_written(self,address):
        if address not in self.state_versions:raise ValueError('Unknown state dependency')
        self.state_versions[address]+=1
    def stamp(self,credit,parameter_reads=(),state_reads=(),target_id=''):
        # The target bank is always included; canonical read sets deduplicate
        # shared ownership. No implicit scan through unrelated banks.
        names=sorted(set(parameter_reads)|{credit.bank});states=sorted(set(state_reads))
        if credit.version!=self.banks.versions[credit.bank]:raise ValueError('Credit already stale at issue')
        return DependencyCreditPacket(credit,tuple((name,self.banks.versions[name]) for name in names),
            tuple((name,self.state_versions[name]) for name in states),target_id)
    def apply(self,packet):
        try:
            parameters=dict(packet.parameter_versions);states=dict(packet.state_versions)
            if len(parameters)!=len(packet.parameter_versions) or len(states)!=len(packet.state_versions):
                raise ValueError('Duplicate dependency address')
            if parameters.get(packet.credit.bank)!=packet.credit.version:
                raise ValueError('Target bank stamp missing or inconsistent')
            for name,version in packet.parameter_versions:
                if name not in self.banks.versions or self.banks.versions[name]!=version:
                    raise ValueError('Parameter dependency moved')
            for name,version in packet.state_versions:
                if name not in self.state_versions or self.state_versions[name]!=version:
                    raise ValueError('State dependency moved')
            receipt=self.banks.apply(packet.credit)
        except (ValueError,KeyError):
            self.rejected+=1;raise
        self.accepted+=1
        return dict(**receipt,target_id=packet.target_id,parameter_dependencies=len(parameters),state_dependencies=len(states))
    @staticmethod
    def projected_metadata_bytes(packet,address_bytes=4,version_bytes=8,event_id_bytes=8):
        # Declared binary layout, not Python object size or measured wire work:
        # target address/version, event/target IDs, two32-bit counts, read pairs.
        pairs=len(packet.parameter_versions)+len(packet.state_versions)
        return address_bytes+version_bytes+2*event_id_bytes+8+pairs*(address_bytes+version_bytes)
