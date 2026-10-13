"""R1 sparse dependency/version contracts for local packet delivery."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sleeping_machines.learning.local_credit_banks import CreditPacket,LocalCreditBanks
from sleeping_machines.learning.dependency_credit_packets import DependencyBankDelivery


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    values={name:torch.nn.Parameter(torch.ones(2)) for name in ('target','producer','unrelated')}
    banks=LocalCreditBanks(values.items(),lr=.003);delivery=DependencyBankDelivery(banks)
    delivery.register_state('memory');delivery.register_state('other_memory')
    def issue():return delivery.stamp(CreditPacket('target',banks.versions['target'],torch.ones(2),'forecast0'),['producer'],['memory'],'fixed_credit_target')
    old=issue();banks.apply(CreditPacket('producer',0,torch.ones(2),'producer_update'))
    snapshot=values['target'].detach().clone()
    try:delivery.apply(old)
    except ValueError:parameter_dependency_rejected=True
    else:raise AssertionError('Dependent producer update accepted as current')
    assert torch.equal(values['target'],snapshot) and banks.versions['target']==0
    old=issue();delivery.state_written('memory')
    try:delivery.apply(old)
    except ValueError:state_dependency_rejected=True
    else:raise AssertionError('Changed observed memory accepted as current')
    fresh=issue();delivery.state_written('other_memory');banks.apply(CreditPacket('unrelated',0,torch.ones(2),'independent'))
    receipt=delivery.apply(fresh);assert banks.versions['target']==1
    assert receipt['parameter_dependencies']==2 and receipt['state_dependencies']==1
    try:delivery.apply(fresh)
    except ValueError:replay_rejected=True
    else:raise AssertionError('Packet replay accepted')
    paths=('experiments/credit/check_dependency_credit_packets.py','sleeping_machines/learning/dependency_credit_packets.py','sleeping_machines/learning/local_credit_banks.py')
    result=dict(status='completed',battle='R1',parameter_dependency_rejected=parameter_dependency_rejected,
        state_dependency_rejected=state_dependency_rejected,unrelated_bank_and_state_progress_allowed=True,
        replay_rejected=replay_rejected,projected_metadata_bytes=delivery.projected_metadata_bytes(fresh),
        gradient_payload_bytes=fresh.credit.gradient.numel()*fresh.credit.gradient.element_size(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
        scope='Serialized event-delivery metadata contract, not distributed atomic validation: reject moved declared parameter/state dependencies even when target bank unchanged; unrelated declared-independent updates do not block delivery. No global epoch, no stale-gradient transport. Read-set completeness must be established by actual credit generation; this fixture declares it and is not an integrated model or quality proof. Metadata assumes32-bit bank/state addresses,64-bit versions/event IDs; framing/routing/handshake additional. Extra dependency checks and bytes are charged, not free locality.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
