"""Actual selection-wrapper trajectory parity, within one guarded job."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import token_language as selected
tag = 'curie_parameterized_token_parity_20261005_v1'
sys.argv = ['selected_token_language_lab.py', '--tag', tag,
    '--readout-init', 'frequency', '--input-init', 'balanced', '--steps', '6',
    '--eval-every', '3', '--train-tokens', '512', '--dev-tokens', '128',
    '--payload', '8', '--heads', '1', '--depth', '2', '--pool', '2',
    '--lanes', '2', '--eval-lanes', '2', '--chunk', '4', '--future-every', '1',
    '--future-score-positions', '2', '--future-site','uniform','--minimum-tail-width','0']
selected.main()
folder = ROOT / 'experiments/results/token_language'
a = torch.load(folder / 'curie_event_resume_full_20261005_v1.pt', weights_only=False)
b = torch.load(folder / (tag + '.pt'), weights_only=False)
def equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b
for key in ('model', 'optimizer', 'state', 'generator', 'alternative_generator',
            'local_generator', 'position_generator','site_generator', 'torch_rng', 'cursor',
            'step', 'total_presentations', 'writes'):
    equal(a[key], b[key])
record = json.loads((folder / (tag + '.selection.json')).read_text())
assert record['selected']['dev_nll'] == min(r['dev_nll'] for r in b['curve'])
assert Path(record['selected']['checkpoint']).exists()
result = dict(status='completed', exact_learning_trajectory=True,
              selected_includes_initial=True, targets=48,
              scope='Actual selection wrapper vs completed event-coverage driver trajectory; initialization preserved as a model artifact')
p = ROOT / 'experiments/results/diagnostics' / (tag + '.json')
if p.exists(): raise FileExistsError(p)
p.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)
