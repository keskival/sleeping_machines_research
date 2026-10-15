import numpy as np
import pytest
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs,swap_observed_halves


def test_all_pairs_use_existing_train_inputs_and_whole_labels_only():
    rows=[r.model_input() for r in corpus() if r.split=='train']
    pairs=training_pairs(rows);assert len(pairs)==64
    assert sorted(i for pair in pairs for i in pair)==list(range(128))
    for i,j in pairs:
        assert rows[i][1]!=rows[j][1]
        assert np.array_equal(swap_observed_halves(rows[i][0]),rows[j][0])
        assert np.array_equal(swap_observed_halves(rows[j][0]),rows[i][0])


def test_pairs_cannot_import_an_unavailable_scene_or_label():
    rows=[r.model_input() for r in corpus() if r.split=='train']
    with pytest.raises(ValueError):training_pairs(rows[:-1])
    events,_=rows[0];assert np.array_equal(events[:,0],swap_observed_halves(events)[:,0])
