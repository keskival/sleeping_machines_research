from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.paired_composition_decision import observed_groups


def test_uncertainty_groups_use_public_geometry_and_cover_correlated_views():
    rows=[r.model_input() for r in corpus() if r.split=='dev']
    groups=observed_groups(rows)
    assert len(groups)==16 and all(len(g)==4 for g in groups)
    assert sorted(i for group in groups for i in group)==list(range(64))
