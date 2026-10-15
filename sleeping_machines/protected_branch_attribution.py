"""Frozen-model branch-history intervention, not a training backend."""
import math
import torch


def without_branch_memory(real):
    """Erase only prior added-stage memory contents before each observed token.

    Parent memories, arrival/seen metadata, protected feedback and clock tapes
    are retained. Splits are causal and keep the original event order. This is
    a content intervention, not a retrained control or an identity-stage bypass.
    """
    def features(core, tokens, state=None, generator=None, **kwargs):
        if kwargs.get('route_credit', 'none') != 'none':
            raise ValueError('Frozen factual attribution only')
        if kwargs.get('force_site') is not None or kwargs.get('suppress_site') is not None:
            raise ValueError('Do not combine route interventions with history erasure')
        if not 0 < core.parent_depth < core.depth:
            raise ValueError('Protected temporal branch required')
        values, probabilities, winners, arrivals = [], [], [], []
        writes = None
        content = kwargs.pop('token_content', None)
        for index in range(tokens.shape[1]):
            if state is not None:
                state = dict(state)
                state['mem'] = [value if depth < core.parent_depth else torch.zeros_like(value)
                                for depth, value in enumerate(state['mem'])]
            local = dict(kwargs)
            if content is not None:
                local['token_content'] = content[index:index+1]
            value, state, pi = real(core, tokens[:, index:index+1], state, generator, **local)
            values.append(value)
            probabilities.extend(pi)
            winners.append(state['race_winners'])
            arrivals.append(state['query_arrivals'])
            writes = state['last_writes'] if writes is None else writes + state['last_writes']
        state['race_winners'] = torch.cat(winners, dim=0)
        state['query_arrivals'] = torch.cat(arrivals, dim=1)
        state['last_writes'] = writes
        return torch.cat(values, dim=1), state, probabilities
    return features


def risk_decomposition(continued_parent, grown, grown_without_correction):
    """Exact arithmetic identity; components are frozen interventions, not causal fitting effects."""
    result = dict(total_excess=grown-continued_parent,
        correction_effect=grown-grown_without_correction,
        fitted_parent_path_excess=grown_without_correction-continued_parent)
    if abs(result['total_excess']-result['correction_effect']-result['fitted_parent_path_excess']) > 1e-10:
        raise ValueError('Nonfinite or inconsistent risks')
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Finite risks required')
    return result
