"""Completed matched private-bank quality/work comparison; stdlib only."""
import math
try:
    from .aws_token_stage_evidence import stage_row, add_utility
    from .aws_private_bank_policy import replica_policy
except ImportError:
    from aws_token_stage_evidence import stage_row, add_utility
    from aws_private_bank_policy import replica_policy

EXACT_FIELDS={'model','optimizer','state','cursor','step','site_generator','position_generator',
              'local_generator','alternative_generator','generator','torch_rng','writes','total_presentations'}


def compare(small,large):
    """Each arm is (completed raw fit, selection, whole work, utility row).

    This reports one paired seed. It neither admits a new fit nor makes a public
    reference claim. Missing work is an error, never an estimated denominator.
    """
    rows=[];recipes=[];sources=[]
    for result,selection,work,utility in (small,large):
        if work is None:raise ValueError('Complete work required for bank comparison')
        row=add_utility(stage_row(result,selection,work),utility)
        args=result['args'];recipe=result.get('bank_recipe',{})
        if recipe.get('bank_policy')!=replica_policy(4,args['pool']):raise ValueError('Declared replica recipe required')
        if utility.get('bank_recipe')!=recipe or work.get('bank_recipe')!=recipe:
            raise ValueError('Utility/work bank recipe mismatch')
        if work.get('final_numeric_training_state_exact') is not True or set(work.get('exact_checkpoint_fields',[]))!=EXACT_FIELDS:
            raise ValueError('Complete numeric training-state replay required')
        fit_sources=result.get('source_sha256',{})
        if not fit_sources or any(work.get('source_sha256',{}).get(k)!=v for k,v in fit_sources.items()):
            raise ValueError('Fit/work source binding mismatch')
        if not args['tie_pools'] or args['temperature']!=1. or args['free_bias']!=0.:
            raise ValueError('Shared-rule controlled-clock comparison required')
        if not all(math.isfinite(row[k]) for k in ('selected_dev_nll','whole_fit_gflops','fitting_mflops_per_target','inference_mflops_per_target')):
            raise ValueError('Finite completed quality/work required')
        rows.append(row);recipes.append(recipe);sources.append(fit_sources)
    a,b=rows
    if (a['pool'],b['pool'])!=(4,16):raise ValueError('Declared U4/U16 pair required')
    configs=[{k:v for k,v in arm[0]['args'].items() if k not in ('tag','pool')} for arm in (small,large)]
    if configs[0]!=configs[1] or a['data_identity']!=b['data_identity'] or sources[0]!=sources[1]:
        raise ValueError('Paired data/settings/source identity required')
    if recipes[0]['source_sha256']!=recipes[1]['source_sha256']:
        raise ValueError('Bank constructor sources differ')
    if a['fitting_targets']!=b['fitting_targets'] or a['selected_writes_per_target']!=b['selected_writes_per_target']:
        raise ValueError('Fitting exposure or selected activity differs')
    gain=a['selected_dev_nll']-b['selected_dev_nll']
    fit_ratio=b['whole_fit_gflops']/a['whole_fit_gflops']
    inference_ratio=b['inference_mflops_per_target']/a['inference_mflops_per_target']
    return dict(status='completed',rows=rows,seed=a['seed'],
        larger_bank_dev_nll_gain=gain,quality_verdict='win' if gain>0 else 'loss' if gain<0 else 'tie',
        whole_fit_work_ratio=fit_ratio,inference_work_ratio=inference_ratio,
        state_capacity_ratio=b['persistent_memory_scalars_per_lane']/a['persistent_memory_scalars_per_lane'],
        scored_key_ratio=b['scored_keys_per_target']/a['scored_keys_per_target'],
        selected_write_ratio=b['selected_writes_per_target']/a['selected_writes_per_target'],
        pareto_win=gain>0 and fit_ratio<=1. and inference_ratio<=1.,
        scope='One paired seed, completed FineWeb GPT-2 development quality and whole-fit/selected DEV arithmetic. Initialization is eligible; entire fit is charged. Memory erasure also clears arrival/seen metadata. Random sampling work unquantified; special functions separate. No public Transformer score, independent-seed confirmation or scaling admission.')
