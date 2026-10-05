"""Admit a reserved larger-data fit only after independent learning and work gates."""
import argparse
import hashlib
import json
from pathlib import Path
from token_capacity_postrepeat_gate import select

ROOT = Path(__file__).resolve().parents[1]


def admit():
    # Existing gate verifies completed repeat/selection/utility and positive learning.
    select('work')
    receipt = json.loads((ROOT / 'experiments/queue/curie_data_growth_64k_repeat_admission_20261005_v1.json').read_text())
    width = receipt['selected_payload']
    audit = json.loads((ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_64k_p{width}_work_20261005_v1.json').read_text())
    assert audit['status'] == 'completed' and audit['fitting_targets'] == 131056
    assert audit['inference_targets'] == 2040 and audit['optimizer_updates'] == 256
    assert audit['curve_parity_max_error'] < 2e-6
    assert all(audit[k]['formula_coverage_complete'] for k in ('fitting', 'inference'))
    assert audit['source_sha256'] == hashlib.sha256((ROOT / 'experiments/token_stage_work.py').read_bytes()).hexdigest()
    parent_tag = f'curie_data_growth_tokens_64k_b64_c16_p{width}_s6_20261005_v1'
    assert Path(audit['control']).name == parent_tag+'.json'
    parent = json.loads((ROOT / f'experiments/results/token_language/{parent_tag}.json').read_text())
    for name, digest in parent['identity']['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    tag = parent_tag.replace('_64k_', '_256k_')
    queue = ROOT / f'experiments/queue/{tag}.txt'
    words = queue.read_text().split()
    assert words[0] == tag
    for flag, value in [('--payload',width),('--seed',6),('--train-tokens',262144),('--steps',1024),('--eval-every',256)]:
        assert words[words.index(flag)+1] == str(value)
    return dict(status='admitted', queue=str(queue.relative_to(ROOT)), payload=width,
                parent=parent_tag, train_tokens=262144, fitting_targets=524272,
                gate='Independent64Klearning/utility plus completed whole-fit/inference accounting and parity/source pins',
                scope='First reserved larger-data cell of the selected width; two passes/four evaluation checkpoints and unchanged core mechanisms. Not a scaling-law fit or public benchmark result.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    receipt=admit()
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['queue'])
