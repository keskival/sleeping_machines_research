"""Select one prepared diagnostic only for the completed admitted256Kfit."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def select():
    receipt=json.loads((ROOT/'experiments/queue/curie_data_growth_256k_admission_20261005_v1.json').read_text())
    assert receipt['status']=='admitted'
    width=receipt['payload'];assert width in (16,24,32)
    tag=f'curie_data_growth_tokens_256k_b64_c16_p{width}_s6_20261005_v1'
    path=ROOT/f'experiments/results/token_language/{tag}.json'
    result=json.loads(path.read_text())
    selected=json.loads(path.with_suffix('.selection.json').read_text())
    assert result['status']==selected['status']=='completed'
    assert result['presentations_total']==524272
    assert result['args']['train_tokens']==262144 and result['args']['payload']==width
    assert abs(selected['selected']['dev_nll']-min(x['dev_nll'] for x in result['curve']))<2e-6
    return f'experiments/queue/curie_data_growth_tokens_256k_p{width}_utility_20261005_v1.txt'


if __name__=='__main__':print(select())
