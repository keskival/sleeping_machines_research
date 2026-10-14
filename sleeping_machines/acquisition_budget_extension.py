"""Explicit budget-only checkpoint migration; exact-resume rules stay intact."""
import copy


def extend_acquisition_checkpoint(checkpoint, completed_packet, steps):
    if completed_packet.get('status')!='completed':
        raise ValueError('Only a completed acquisition can be extended')
    previous=checkpoint['settings']['steps']
    if not isinstance(steps,int) or isinstance(steps,bool) or steps<=previous:
        raise ValueError('Strictly larger integer target-update budget required')
    if checkpoint['step']!=previous or completed_packet['args']['steps']!=previous:
        raise ValueError('Completed packet and checkpoint budgets disagree')
    if checkpoint['identity']!=completed_packet['identity']:
        raise ValueError('Completed packet and checkpoint identity disagree')
    recorded={k:v for k,v in completed_packet['args'].items()
              if k not in ('tag','resume','stop_after_step')}
    if recorded!=checkpoint['settings']:
        raise ValueError('Completed packet and checkpoint settings disagree')
    if checkpoint['settings']['exposure']!='fresh':
        raise ValueError('This migration extends fresh-token acquisition only')
    # Only the planned endpoint changes. Preserve every tensor and trajectory
    # receipt, not just weights. Parent objects must remain untouched.
    result=copy.deepcopy(checkpoint)
    result['settings']['steps']=steps
    return result
