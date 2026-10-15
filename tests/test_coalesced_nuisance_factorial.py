import numpy as np
from experiments.hierarchy.coalesced_nuisance_factorial import factorial


def test_factorial_separates_coordinate_and_timing_changes():
    a,translation,timing,both=factorial(1,2,0)
    assert all(s.label==a.label for s in [translation,timing,both])
    assert np.array_equal(a.events[:,0],translation.events[:,0])
    # Point/polarity bags are identical for a timing-only intervention.
    def bag(scene):return sorted(map(tuple,scene.events[:,1:]))
    assert bag(a)==bag(timing) and bag(translation)==bag(both)
    assert np.array_equal(timing.events[:,0],both.events[:,0])
