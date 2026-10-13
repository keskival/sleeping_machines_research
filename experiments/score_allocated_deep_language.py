"""R1 completed native deep fit: exact checkpoint replay and long-history DEV."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import allocated_depth_token_language as driver
from public_token_scoring import score
from sleeping_machines.connected_depth_risk import check_cross_audit_contract


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--run',required=True)
    a=parser.parse_args();driver.torch.set_num_threads(1);begin=time.monotonic()
    check_cross_audit_contract()
    base=ROOT/f'experiments/results/token_language/{a.run}'
    packet=json.loads(base.with_suffix('.json').read_text())
    assert packet['status']=='completed'
    for name,digest in packet['source_sha256'].items():assert driver.sha(ROOT/name)==digest,name
    args=SimpleNamespace(**packet['args'])
    assert args.depth>=8
    train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file)
    counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
    order=driver.torch.argsort(counts,descending=True,stable=True)
    driver.torch.manual_seed(args.seed)
    model=driver.Model(args,order,counts)
    saved=driver.torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False)
    assert saved['step']==args.steps
    model.load_state_dict(saved['model']);del saved
    replay=driver.evaluate(model,driver.interval_tensor(dev,args.dev_offset,args.dev_tokens,args.eval_lanes),args)
    assert replay==packet['curve'][-1]['dev_nll']
    long_history=score(model,dev,args,driver,25165824,8192,1024,8)
    # Each cut removes only the incoming content projection at this stage.
    # Keep its bias, clocks, private bank and arrival inputs. This is not a
    # layer deletion or a shallower trained-model comparator.
    paths=[]
    for layer,mix in enumerate(model.core.channel_mix):
        original=mix.weight.detach().clone()
        try:
            with driver.torch.no_grad():mix.weight.zero_()
            cut=score(model,dev,args,driver,25165824,8192,1024,8)
        finally:
            with driver.torch.no_grad():mix.weight.copy_(original)
        assert cut['targets']==long_history['targets']
        paths.append(dict(layer=layer,cut_nll=cut['nll'],
                          nll_increase=cut['nll']-long_history['nll'],
                          removed_projection_weight_norm=float(original.norm())))
    restored=score(model,dev,args,driver,25165824,8192,1024,8)
    assert restored==long_history,'Depth-path diagnostic failed to restore exact model/tape'
    result=dict(status='completed',battle='R1',run=a.run,depth=args.depth,
                parameters=packet['parameters'],targets=packet['presentations_total'],
                final_development_nll=replay,long_history_dev=long_history,
                incoming_content_path_cuts=paths,restored_score_parity=True,
                cross_audit_risk_math_contract='passed depth8/16 finite-sum and parameter-gradient parity; no integrated credit fit',
                checkpoint_replay_targets=args.eval_lanes*(args.dev_tokens//args.eval_lanes-1),
                long_history_diagnostic_scored_targets=8192*(args.depth+2),
                diagnostic_scored_targets=8192*(args.depth+2)+args.eval_lanes*(args.dev_tokens//args.eval_lanes-1),
                successful_trajectory_training_s=packet['train_wall_s_total'],
                successful_trajectory_training_s_per_target=packet['train_wall_s_total']/packet['presentations_total'],
                whole_fit_process_wall_s=packet.get('whole_fit_process_wall_s',packet['process_wall_s']),
                whole_fit_process_s_per_target=packet.get('whole_fit_process_wall_s',packet['process_wall_s'])/packet['presentations_total'],
                interruption_work_scope=packet.get('whole_fit_work_scope','Single process; includes setup, fitting, checkpointing and DEV evaluation.'),
                selected_writes_per_target=args.depth*args.heads,
                scored_keys_per_target=args.depth*args.heads*args.pool,
                teacher_work=packet['teacher_work'],allocation=packet['depth_credit_allocation'],
                input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in (base.with_suffix('.json'),base.with_suffix('.pt'))},
                source_sha256={name:driver.sha(ROOT/name) for name in ('experiments/score_allocated_deep_language.py','experiments/public_token_scoring.py','sleeping_machines/connected_depth_risk.py')},
                wall_s=time.monotonic()-begin,
                scope='Single seed, frozen final checkpoint, exact replay and8192 reusedDEV targets with1024 disjoint histories. '
                      'Per-stage frozen incoming-content cuts retain bias/clocks/banks/arrival inputs and share issued random tapes; '
                      'they measure dependence under intervention, not a trained shallower comparison or proof of optimal depth. '
                      'No public validation or external score. Full factual windowBP/Adam/allocator/teacher work paid; no ASIC energy claim.')
    path=ROOT/f'experiments/results/diagnostics/{a.tag}.json';path.parent.mkdir(exist_ok=True)
    with path.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('allocation','input_sha256','source_sha256')}),flush=True)


if __name__=='__main__':main()
