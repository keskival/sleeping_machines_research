"""Check the full-score endpoint against the completed original trajectory."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import sampled_utility_token_language_lab as lab
tag = 'curie_sampled_fullscore_parity_20261005_v1'
sys.argv = ['sampled_utility_token_language_lab.py', '--tag', tag,
    '--readout-init', 'frequency', '--input-init', 'balanced', '--steps', '6',
    '--eval-every', '3', '--train-tokens', '512', '--dev-tokens', '128',
    '--payload', '8', '--heads', '1', '--depth', '2', '--pool', '2',
    '--lanes', '2', '--eval-lanes', '2', '--chunk', '4', '--future-every', '1',
    '--future-score-positions', '0']
lab.main()
folder = ROOT / 'experiments/results/token_language'
old = torch.load(folder / 'curie_integrated_resume_full_20261005_v1.pt', weights_only=False)
new = torch.load(folder / (tag + '.pt'), weights_only=False)
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
            'local_generator', 'torch_rng', 'cursor', 'step', 'total_presentations', 'writes'):
    equal(old[key], new[key])
for a, b in zip(old['curve'], new['curve']):
    for key in ('step', 'train_nll', 'dev_nll'): equal(a[key], b[key])
result = dict(status='completed', contract='Actual full-score sampled-driver endpoint equals original 6-update learning trajectory',
              exact=True, targets=48, scope='Tiny seed6 learnability fixture; no generalization claim')
p = ROOT / 'experiments/results/diagnostics' / (tag + '.json')
if p.exists(): raise FileExistsError(p)
p.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
