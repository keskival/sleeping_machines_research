"""Causal learned document relevance in the existing normalized native law.

A zero-initialized pair of log-rate biases preserves the old forward law.
No memory is reset; useful cross-document causes may retain/increase rate.
"""
import torch
from torch import nn
from .native_indexed_cause_head import indexed_cause_nll as original_cause_law


class BoundaryHeadView:
    def __init__(self,head,same_document):
        self.head=head
        cross=(~same_document).to(head.document_bias.dtype)
        self.b_rep=head.b_rep+head.document_bias[0]*cross
        self.b_cont=head.b_cont+head.document_bias[1]*cross
    def __getattr__(self,name):return getattr(self.head,name)


def document_cause_law(head,views,target,candidates,**kwargs):
    if not hasattr(head,'document_bias'):
        return original_cause_law(head,views,target,candidates,**kwargs)
    same=candidates['same_document']
    if same.dtype!=torch.bool or same.shape!=candidates['tokens'].shape:
        raise ValueError('one causal document-membership bit per stored event required')
    return original_cause_law(BoundaryHeadView(head,same),views,target,candidates,**kwargs)


def enable_document_relevance(model,optimizer):
    if hasattr(model.causes,'document_bias'):raise ValueError('document law already enabled')
    model.causes.register_parameter('document_bias',nn.Parameter(model.causes.b_rep.new_zeros(2)))
    optimizer.param_groups[0]['params'].append(model.causes.document_bias)
    # Explicit process-local dispatch used by the native stream. Models without
    # the new parameter retain the original law; source files remain unchanged.
    import experiments.r1_local_native_deep as canonical
    canonical.indexed_cause_nll=document_cause_law
    # CLI drivers historically import this module with its short name.
    import sys
    if 'r1_local_native_deep' in sys.modules:sys.modules['r1_local_native_deep'].indexed_cause_nll=document_cause_law
    return dict(new_scalars=2,initial_bias=[0.,0.],memory_reset=False,
        semantics='Add learned cross-document log-rate biases to repeat/continuation causes;all states/history and normalized word/cause laws retained.')
