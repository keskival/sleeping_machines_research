"""Read completed token fits; diagnose exposure without executing a model."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    engine = ROOT / 'experiments/horizon_token_language_engine.py'
    source = engine.read_text()
    for contract in ('probe = interval_tensor(tr,0,min(8192,a.train_tokens),a.eval_lanes)',
                     'train_nll=evaluate(model,probe[:,:min(129,probe.shape[1])],a)',
                     'clip_grad_norm_(model.parameters(),1.)'):
        assert contract in source, contract
    rows = []
    for budget, label in ((65536, '64k'), (262144, '256k')):
        for payload, seed in ((16,6),(24,6),(24,7),(32,6)):
            path = ROOT / f'experiments/results/token_language/curie_data_growth_tokens_{label}_b64_c16_p{payload}_s{seed}_20261005_v1.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result['status'] != 'completed':
                continue
            args = result['args']
            assert (args['train_tokens'], args['payload'], args['seed']) == (budget,payload,seed)
            assert digest(engine) == result['identity']['source_sha256']['experiments/horizon_token_language_engine.py']
            assert args['lr'] == .003 and args['weight_decay'] == .01
            curve = result['curve']
            selected = min(curve, key=lambda row: row['dev_nll'])
            final = curve[-1]
            logged = [row for row in curve if 'gradient_norm' in row]
            rows.append(dict(train_tokens=budget,payload=payload,seed=seed,
                result_path=str(path.relative_to(ROOT)),result_sha256=digest(path),
                selected_step=selected['step'],selected_nll=selected['dev_nll'],
                final_nll=final['dev_nll'],final_minus_selected_nll=final['dev_nll']-selected['dev_nll'],
                train_probe_selected_nll=selected['train_nll'],train_probe_final_nll=final['train_nll'],
                train_probe_final_minus_selected_nll=final['train_nll']-selected['train_nll'],
                selected_route_entropy=selected.get('route_entropy'),final_route_entropy=final.get('route_entropy'),
                logged_gradient_norms=[row['gradient_norm'] for row in logged],
                logged_above_clip_threshold=sum(row['gradient_norm']>1 for row in logged),
                logged_gradient_checkpoints=len(logged),
                probe_scored_targets=128*args['eval_lanes']))
    return dict(status='completed_read_only_diagnostic',rows=rows,
        producer_sha256=digest(Path(__file__)),engine_sha256=digest(engine),
        scope='Development-only trajectories. train_nll is a fixed fresh-state TRAIN probe of 1024 targets, not full-training loss. Gradient norms are pre-clipping at four logged checkpoints, not every update. Route entropy is factual race probability entropy; lower entropy does not by itself demonstrate receiver collapse or explain loss.',
        next_test='Keep admitted 256K fit unchanged. If P24 final-minus-selected development NLL exceeds 0.05 again while TRAIN probe improves, compare one lower constant learning rate 0.001 against saved 0.003 at the same data, initialization, seed, two-pass exposure and checkpoint cadence before another scale increase. Preserve all mechanisms. No candidate fit admitted by this diagnostic; complete fitting work must be audited independently.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    record=audit();output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(rows=len(record['rows']),declining_development=sum(row['final_minus_selected_nll']>0 for row in record['rows']))))
