import io
import pytest
import torch
from sleeping_machines.native_cause_bank import NativeCauseBank


def test_successors_mature_only_after_observation_and_are_not_retrieval_neighbors():
    bank=NativeCauseBank(2,4,capacity=32)
    for t,token in enumerate([3,7,4,9,3]):bank.append(token,torch.full((2,4),float(t)),[('unit',t%2)],t)
    result=bank.query(3,[('unit',0)],4,recent=0,per_key=8)
    records={int(s):(int(t),int(n)) for s,t,n in zip(result['serials'],result['tokens'],result['successors'])}
    assert records[0]==(3,7) and records[2]==(4,9) and records[4]==(3,-1)
    assert not result['features'].requires_grad


def test_indexed_old_cause_survives_many_credit_batches_without_a_full_scan():
    bank=NativeCauseBank(1,2,capacity=256)
    bank.append(81,torch.ones(1,2),[('concept',99)],0)
    for t in range(1,201):bank.append(2,torch.full((1,2),float(t)),[('other',1)],t)
    before=bank.index_entries_examined
    result=bank.query(7,[('concept',99)],200,recent=2,per_key=1,maximum=4)
    assert 0 in result['serials'].tolist()
    assert bank.index_entries_examined-before<=4
    assert len(result['serials'])<=4


def test_eviction_does_not_alias_an_old_native_binding_to_a_new_record():
    bank=NativeCauseBank(1,2,capacity=3)
    bank.append(11,torch.zeros(1,2),[('old',1)],0)
    for t in range(1,6):bank.append(2,torch.ones(1,2)*t,[('new',1)],t)
    result=bank.query(11,[('old',1)],5,recent=1)
    assert 0 not in result['serials'].tolist() and (result['serials']>=3).all()
    with pytest.raises(ValueError):bank.query(2,[],4)


def test_serialized_numeric_index_state_preserves_queries():
    bank=NativeCauseBank(2,3,capacity=20)
    for t,token in enumerate([1,2,3,1,2]):bank.append(token,torch.ones(2,3)*t,[('unit',t%2)],t)
    buf=io.BytesIO();torch.save(bank,buf);buf.seek(0);restored=torch.load(buf,weights_only=False)
    for key,value in bank.query(2,[('unit',0)],4).items():
        assert torch.equal(value,restored.query(2,[('unit',0)],4)[key])


def test_suffix_matching_uses_only_revealed_content():
    bank=NativeCauseBank(1,2)
    for t,token in enumerate([1,2,3,9,1,2,3]):bank.append(token,torch.zeros(1,2),[],t)
    result=bank.query(3,[],6,token_matches=8)
    rows=dict(zip(result['serials'].tolist(),result['suffix_matches'].tolist()))
    assert rows[2]==3 and rows[6]==7


def test_unqueried_index_metadata_is_bounded_by_resident_capacity():
    bank=NativeCauseBank(1,2,capacity=7)
    for t in range(300):bank.append(t,torch.zeros(1,2),[('never_reused',t)],t)
    assert bank.storage()['index_references']==14
    assert bank.storage()['index_keys']==14
