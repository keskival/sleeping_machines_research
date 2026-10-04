"""Mean-normalized clock-noise control for a hard exponential race (stdlib).

The winner is untouched. Only the common first clock is reshaped. Mean refers
to the unbounded clock, before the native .001+.010*T/(1+T) arrival transform.
"""
import math


def check_temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Clock temperature must be finite in [0, 1]')
    return float(value)


def log_normalizer(scores):
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError('Nonempty finite scores required')
    high = max(scores)
    return high + math.log(sum(math.exp(s - high) for s in scores))


def clock_from_unit_noise(scores, unit_exponential, temperature):
    temperature = check_temperature(temperature)
    if not math.isfinite(unit_exponential) or unit_exponential <= 0:
        raise ValueError('Positive finite unit exponential required')
    return math.exp(temperature * math.log(unit_exponential) - math.lgamma(1 + temperature) - log_normalizer(scores))


def race(scores, exponential_noise, temperature=1.):
    temperature = check_temperature(temperature)
    if len(scores) != len(exponential_noise) or not scores:
        raise ValueError('Matched nonempty scores/noise required')
    if not all(math.isfinite(e) and e > 0 for e in exponential_noise):
        raise ValueError('Positive finite candidate noise required')
    logz = log_normalizer(scores)
    logtimes = [math.log(e) - s for e, s in zip(exponential_noise, scores)]
    winner = min(range(len(scores)), key=logtimes.__getitem__)
    logw = logtimes[winner] + logz
    first = math.exp(temperature * logw - math.lgamma(1 + temperature) - logz)
    return winner, first


def moments(scores, temperature):
    temperature = check_temperature(temperature)
    mean = math.exp(-log_normalizer(scores))
    cv2 = max(0., math.exp(math.lgamma(1 + 2 * temperature) - 2 * math.lgamma(1 + temperature)) - 1)
    return dict(mean_unbounded_clock=mean, variance_unbounded_clock=cv2 * mean * mean,
                coefficient_of_variation=math.sqrt(cv2))


def factorized_clock_credit(scores, first):
    """Derivative with the independent unit-exponential variable held fixed."""
    logz = log_normalizer(scores)
    return [-first * math.exp(s - logz) for s in scores]
