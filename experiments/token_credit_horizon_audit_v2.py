"""Frozen actual-write consequences beyond the current16-token credit window."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT))
import event_coverage_token_language_lab as lab


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.set_num_threads(1)
record_path = ROOT / 'experiments/results/token_language/curie_event_credit_2k_k4_20261005_v1.json'
record = json.loads(record_path.read_text())
args = SimpleNamespace(**record['args'])
data = lab.load_tokens(args.train_file)
counts = lab.LaneTokens(data, 0, args.train_tokens, args.lanes).counts()
order = torch.argsort(counts, descending=True, stable=True)
torch.manual_seed(args.seed)
model = lab.Model(args, order, counts)
checkpoint = Path(record['best_checkpoint'])
model.load_state_dict(torch.load(checkpoint, weights_only=True)); model.eval()
stream = lab.interval_tensor(data, 0, args.train_tokens, args.lanes)[:, :129]
ids, targets = stream[:, :-1], stream[:, 1:]
sites = [(0,0,0),(7,0,0),(15,0,0),(7,1,1)]
site_records=[]
with torch.no_grad():
    factual, state, _ = model(ids, None, torch.Generator().manual_seed(7), args)
    base = model.readout.nll(factual, targets).reshape_as(targets)
    for site in sites:
        winner = state['race_winners'][site[0], site[1], :, site[2]]
        alternatives = []
        for offset in range(1, args.pool):
            choice = (winner + offset) % args.pool
            model.force_site = (*site, choice)
            shadow, _, _ = model(ids, None, torch.Generator().manual_seed(7), args)
            assert torch.equal(factual[:, :site[0]], shadow[:, :site[0]])
            delta = model.readout.nll(shadow, targets).reshape_as(targets) - base
            assert torch.equal(delta[:, :site[0]], torch.zeros_like(delta[:, :site[0]]))
            absolute = delta.abs().sum(0)
            total = float(absolute.sum())
            alternatives.append(dict(choice=choice.tolist(), signed_loss_delta=delta.sum(0).tolist(),
                absolute_loss_delta=absolute.tolist(), total_absolute_loss_delta=total,
                beyond_credit_absolute_fraction=float(absolute[16:].sum()) / total if total else 0.,
                within_credit_signed_delta=float(delta[:, :16].sum()),
                beyond_credit_signed_delta=float(delta[:, 16:].sum())))
        site_records.append(dict(site=list(site),winner=winner.tolist(),alternatives=alternatives))
model.force_site = None
result = dict(status='completed', evidence='frozen training-loss intervention diagnostic',
    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
    source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                  'experiments/event_coverage_token_language_lab.py':sha(ROOT/'experiments/event_coverage_token_language_lab.py'),
                  'sleeping_machines/sparse_counterfactual_episodes.py':sha(ROOT/'sleeping_machines/sparse_counterfactual_episodes.py')},
    checkpoint_source_identity=record['identity'], sites=site_records,
    targets=targets.numel(), horizon=128, current_credit=16,
    observed_eos_positions=[(ids[lane]==50256).nonzero().flatten().tolist() for lane in range(args.lanes)],
    scope='Seed6 completed best trained2K member, frozen parameters, same train lanes; actual alternate write with factual first clock/common future race noise. Absolute per-position loss perturbation measures missing consequence coverage, not unbiased route-credit gradient or held-out quality.')
out = ROOT / 'experiments/results/diagnostics/curie_token_credit_horizon_audit_20261005_v2.json'
if out.exists(): raise FileExistsError(out)
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status='completed',sites=[dict(site=r['site'],beyond_credit_absolute_fraction=[a['beyond_credit_absolute_fraction'] for a in r['alternatives']]) for r in site_records])),flush=True)
