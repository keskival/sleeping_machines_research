"""Select prepared diagnostics/work queues from a completed seed-repeat receipt."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select(mode):
    receipt = json.loads((ROOT / 'experiments/queue/curie_data_growth_64k_repeat_admission_20261005_v1.json').read_text())
    assert receipt['status'] == 'admitted'
    width = receipt['selected_payload']
    assert width in (16, 24, 32)
    tag = f'curie_data_growth_tokens_64k_b64_c16_p{width}_s7_20261005_v1'
    result = json.loads((ROOT / f'experiments/results/token_language/{tag}.json').read_text())
    selection = json.loads((ROOT / f'experiments/results/token_language/{tag}.selection.json').read_text())
    assert result['status'] == selection['status'] == 'completed'
    assert result['presentations_total'] == 131056
    if mode == 'utility':
        return f'experiments/queue/curie_data_growth_tokens_64k_p{width}_s7_utility_20261005_v1.txt'
    utility = json.loads((ROOT / f'experiments/results/diagnostics/curie_data_growth_tokens_64k_p{width}_s7_utility_20261005_v1.json').read_text())
    assert utility['status'] == 'completed'
    row = utility['rows'][0]
    assert row['tag'] == tag and row['selected'] == selection['selected']
    assert row['promotion']['small_fit_promotable'] and row['context_gain'] > 0
    assert row['matched_rng'] and row['partition_parity']
    return f'experiments/queue/curie_data_growth_tokens_64k_p{width}_work_20261005_v1.txt'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['utility', 'work'], required=True)
    args = p.parse_args()
    print(select(args.mode))
