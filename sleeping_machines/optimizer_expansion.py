"""Name-based optimizer-state preservation when adding a learning module.

Existing parameters must be numerically identical. Added parameters start with
no optimizer state; retained moments and group settings are copied exactly.
"""
import copy
import torch


def optimizer_for_expanded_model(source_model, source_optimizer, target_model, new_group=0):
    source = dict(source_model.named_parameters())
    target = dict(target_model.named_parameters())
    for name, parameter in source.items():
        if name not in target:
            raise ValueError(f'Missing retained parameter: {name}')
        replacement = target[name]
        if parameter.shape != replacement.shape or parameter.dtype != replacement.dtype:
            raise ValueError(f'Retained parameter shape/dtype changed: {name}')
        if parameter.device != replacement.device or not torch.equal(parameter, replacement):
            raise ValueError(f'Retained parameter device/value changed: {name}')
    groups = source_optimizer.param_groups
    if not 0 <= new_group < len(groups):
        raise ValueError('Invalid optimizer group for added parameters')
    source_names = {id(parameter): name for name, parameter in source.items()}
    replacements = []
    seen = set()
    transferred = []
    for group in groups:
        copied = {key: copy.deepcopy(value) for key, value in group.items() if key != 'params'}
        copied['params'] = []
        for parameter in group['params']:
            name = source_names.get(id(parameter))
            if name is None or name in seen:
                raise ValueError('Optimizer parameter missing from model or duplicated')
            seen.add(name)
            copied['params'].append(target[name])
            transferred.append((parameter, target[name]))
        replacements.append(copied)
    added = [parameter for name, parameter in target.items()
             if name not in source and parameter.requires_grad]
    replacements[new_group]['params'].extend(added)
    optimizer = type(source_optimizer)(replacements, **copy.deepcopy(source_optimizer.defaults))
    for original, replacement in transferred:
        if original in source_optimizer.state:
            optimizer.state[replacement] = copy.deepcopy(source_optimizer.state[original])
    return optimizer
