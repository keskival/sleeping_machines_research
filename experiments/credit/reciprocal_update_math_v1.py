"""B1/R1: non-fitting analytic witnesses for learning the update function."""
import math


def audit():
    # An inner example has a noisy target. An independent future target has the
    # same latent task mean but independent noise. Starting parameter is zero.
    # One update is w' = alpha*y_support. E[query squared error]/2 follows below.
    latent_variance, noise_variance = 1.0, 4.0
    future = lambda a: ((a-1)**2*latent_variance + a*a*noise_variance + noise_variance)/2
    immediate = lambda a: (a-1)**2*(latent_variance+noise_variance)/2
    best = latent_variance/(latent_variance+noise_variance)
    assert best == .2 and future(best) < future(1.) and immediate(1.) < immediate(best)
    h = 1e-5
    analytic = (best-1)*latent_variance+best*noise_variance
    fd = (future(best+h)-future(best-h))/(2*h)
    assert abs(fd-analytic) < 1e-10
    # A shared update map can amortize different curvatures. A single scalar
    # step cannot solve both quadratics from the same initial displacement.
    curvatures = (1., 9.)
    scalar = sum(curvatures)/sum(v*v for v in curvatures)
    scalar_remaining = sum(h*(1-scalar*h)**2/2 for h in curvatures)/2
    structured_remaining = sum(h*(1-(1/h)*h)**2/2 for h in curvatures)/2
    assert scalar_remaining > .1 and structured_remaining == 0.
    # Route exposure changes the learner, not merely this example's output.
    # Recruitment costs .1 now, improves two independent future uses by .2 each.
    recruitment_now, future_gain = -.1, 2*.2
    assert recruitment_now < 0 < recruitment_now+future_gain
    # Preserving complementary options has value only with observable evidence.
    adaptive = (max(1.,-1.)+max(-1.,1.))/2
    commit = max((1.-1.)/2,(-1.+1.)/2)
    return dict(status='completed',battle='B1/R1 reciprocal update learner',
                scope='analytic expectation contracts; no fitting, no learned optimizer performance claim',
                future_optimal_step=best, same_example_optimal_step=1.,
                future_risk_at_future_step=future(best),future_risk_at_same_example_step=future(1.),
                meta_derivative_error=abs(fd-analytic),optimal_shared_scalar_step=scalar,
                shared_scalar_remaining_loss=scalar_remaining,curvature_conditioned_remaining_loss=structured_remaining,
                recruitment_immediate_value=recruitment_now,recruitment_continuation_value=recruitment_now+future_gain,
                information_option_value=adaptive-commit,
                decision='Train update quality on independent continuation; distinguish update geometry, attribution and recruitment')


if __name__ == '__main__':
    import json
    print(json.dumps(audit(),indent=2))
