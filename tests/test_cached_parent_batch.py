import torch
from sleeping_machines.cached_parent_hierarchy import CachedParentHierarchy


def test_suffix_replay_matches_full_for_two_distinct_episodes():
    torch.manual_seed(18)
    model=CachedParentHierarchy(input_width=3,width=4,levels=3,refinements=2,roles=2)
    features=torch.rand(2,3,8,8)
    output=model(features,seed=19)
    for candidate in output['audits'][4]['valid'][0,0].nonzero().flatten().tolist():
        suffix,_=model.suffix_logits(output,4,0,1,candidate)
        full=model(features,seed=19,force=(4,0,1,candidate))
        assert torch.equal(suffix,full['logits'])
