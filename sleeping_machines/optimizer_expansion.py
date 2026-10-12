"""Name-based optimizer-state preservation when adding a learning module.

Existing parameters must be numerically identical. Added parameters start with
no optimizer state; retained moments and group settings are copied exactly.
"""
import copy
import inspect
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
    # Some PyTorch optimizers retain internal defaults absent from their public
    # constructor (AdamW's decoupled_weight_decay, for example). Group options
    # remain copied; pass only supported constructor keywords.
    signature = inspect.signature(type(source_optimizer).__init__)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in signature.parameters.values())
    defaults = {key: copy.deepcopy(value) for key, value in source_optimizer.defaults.items()
                if accepts_kwargs or key in signature.parameters}
    optimizer = type(source_optimizer)(replacements, **defaults)
    for original, replacement in transferred:
        if original in source_optimizer.state:
            optimizer.state[replacement] = copy.deepcopy(source_optimizer.state[original])
    return optimizer
