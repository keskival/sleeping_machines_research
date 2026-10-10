"""R1 continuous-credit finite proofs; no training or model/data access."""
import pytest
from experiments.credit.continuous_residual_credit import estimate,enumerate_audits


def test_shared_parameter_residuals_preserve_mean_and_quadratic_variance():
    baseline=[1.,-.3,2.];corrections=[[.1,.7,-.2],[.4,-.3,.8],[-.6,.2,.5]]
    target=[b+sum(c[j] for c in corrections) for j,b in enumerate(baseline)]
    p=[.2,.4,.7]
    poor=enumerate_audits(baseline,corrections,p,target)
    assert poor['maximum_mean_error']<1e-12
    assert poor['variance_trace']==pytest.approx(poor['declared_variance_trace'],abs=1e-12)
    improved=[b+.75*sum(c[j] for c in corrections) for j,b in enumerate(baseline)]
    better=enumerate_audits(improved,[[.25*x for x in c] for c in corrections],p,target)
    assert better['maximum_mean_error']<1e-12
    assert better['variance_trace']==pytest.approx(poor['variance_trace']/16,abs=1e-12)
    with pytest.raises(ValueError):estimate(baseline,corrections,[0.,.4,.7],[1,0,0])


def test_overlapping_cut_full_jacobians_and_full_adjoints_double_count():
    # x1=theta; x2=theta+x1; loss=x1+x2. True derivative is 3.
    total_producer_jacobians=[1.,2.];original_graph_adjoints=[2.,1.]
    cut_graph_adjoints=[1.,1.];local_parameter_jacobians=[1.,1.]
    assert sum(j*a for j,a in zip(total_producer_jacobians,cut_graph_adjoints))==3.
    assert sum(j*a for j,a in zip(local_parameter_jacobians,original_graph_adjoints))==3.
    assert sum(j*a for j,a in zip(total_producer_jacobians,original_graph_adjoints))==4.
