"""Source-bound arithmetic lower bound for the published dense record; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def derive():
    meta_path=ROOT/'experiments/references/modded_nanogpt_20250126_metadata.json'
    source_path=ROOT/'experiments/references/modded_nanogpt_20250126_0bdd5ee9.txt'
    meta=json.loads(meta_path.read_text());blob=source_path.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==meta['log_sha256']
    source=blob.decode().split('\n'+'='*100)[0]
    # Pin the actual full-vocabulary projection and both explicit backward contractions.
    for text in ('self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128), use_fp8=True',
                 'logits = self.lm_head(x)', 'grad_x = torch._scaled_mm(',
                 'grad_w = torch._scaled_mm(', 'return grad_x, grad_w, None, None, None',
                 'model(input_seq, target_seq, sw_num_blks(window_size)).backward()',
                 'model_dim=768', 'vocab_size=50257', 'num_iterations = 1770'):
        assert text in source,text
    width=meta['model_width'];vocab=((meta['vocabulary']+127)//128)*128
    targets=meta['world_size']*meta['sequence_length']*meta['iterations']
    assert targets==meta['training_token_presentations']
    forward=2*width*vocab
    fitting=3*forward
    return dict(status='completed_source_bound_lower_bound',complete_reference_work=False,
                reference=meta['reference'],source_url=meta['source_url'],source_sha256=meta['log_sha256'],
                width=width,padded_output_classes=vocab,mac_flops=2,
                fitting_targets=targets,inference_targets=meta['validation_tokens'],
                fitting_head_arithmetic_flops=targets*fitting,
                fitting_head_arithmetic_flops_per_target=fitting,
                inference_head_arithmetic_flops_per_target=forward,
                inference_head_arithmetic_flops=meta['validation_tokens']*forward,
                formula='Forward XW^T and explicit backward dY W / dY^T X: each 2*targets*768*50304; training three contractions, evaluation one.',
                included='Dense output projection contractions only, source-derived conventional arithmetic.',
                excluded='Attention, local MLPs, embeddings, scalar operations, normalization/softcap/cross entropy, dtype scaling/casts, optimizer, communication, diagnostics and initialization. All excluded costs are nonnegative; no complete baseline FLOP claim.',
                scope='Arithmetic lower bound, not an executed trace, GPU timing, traffic or energy. FP8 training and FP32/BF16 evaluation still count two arithmetic FLOPs per multiply-add; bit precision/hardware cost differs from CPU native. Published NLL3.2774 belongs to the full reference/public population, not this isolated head. Unequal native development data/quality prevents a benchmark win claim.',
                producer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    result=derive();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('fitting_targets','fitting_head_arithmetic_flops','fitting_head_arithmetic_flops_per_target','inference_head_arithmetic_flops_per_target','complete_reference_work')}))
