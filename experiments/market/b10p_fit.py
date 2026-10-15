"""B10-P fit: warm-start an acquired B10 tape model, add causal price inputs and the tick-cell horizon-return head, train on
the Aug 1-24 TRAIN windows and select by DEV (Aug 25-31) return likelihood. Fixed-weight development comparison."""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/market'))
from b10_depth_tpp_v1 import TRAIN, VAL, CTX
from b10p_data import split_windows, standardize, label_scales, receipt, HORIZONS
from b10p_models import PriceRace

ACQUIRED = {f: f'experiments/results/market/aws_b10_stable_{f}_s0_20261014_v1.json' for f in ('native', 'lstm', 'transformer')}
SOURCES = ['experiments/market/b10p_fit.py', 'experiments/market/b10p_models.py', 'experiments/market/b10p_data.py',
           'experiments/market/b10_matched_models.py', 'experiments/market/b10_depth_tpp_v1.py', 'experiments/market/b10_stable_clock_head.py',
           'experiments/tpp/race_tpp_v5.py']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stack(rows, keys=('t', 'mark', 'feats', 'lo', 'hi')):
    return [torch.tensor(np.stack([r[k] for r in rows])) for k in keys]


@torch.no_grad()
def evaluate(model, dev, bs=16):
    model.eval(); tape = 0.; n = 0; price = np.zeros(len(HORIZONS)); count = np.zeros(len(HORIZONS))
    for b in range(0, len(dev), bs):
        t, m, f, lo, hi = stack(dev[b:b+bs]); tl, ml, pl, v = model.all_terms(t, m, f, lo, hi)
        tape += float((tl+ml)[:, CTX-1:].sum()); n += tl[:, CTX-1:].numel()
        price += pl[:, CTX:].sum((0, 1)).numpy(); count += v[:, CTX:].sum((0, 1)).numpy()
    model.train()
    per = (-price/count).tolist()
    return dict(tape_nll=-tape/n, tape_targets=n, return_nll=dict(zip([f'{int(h)}s' for h in HORIZONS], per)),
                return_targets=count.astype(int).tolist(), return_nll_sum=float(sum(per)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--family', choices=list(ACQUIRED), required=True)
    p.add_argument('--epochs', type=int, required=True); p.add_argument('--lr', type=float, default=5e-4); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-batches', type=int, default=0, help='smoke: stop each epoch after this many batches')
    a = p.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64); began = time.monotonic()
    out = ROOT/f'experiments/results/market/{a.tag}.json'; assert not out.exists(); progress = out.with_suffix('.progress.pt'); best_path = out.with_suffix('.best.pt')
    train = split_windows(TRAIN, 20); dev = split_windows(VAL, 10); assert len(train) == 801 and len(dev) == 756
    moments = standardize(train, dev); scales = label_scales(train)
    gaps = np.concatenate([np.diff(r['t']) for r in train]); gaps = gaps[gaps > 0]
    torch.manual_seed(a.seed); model = PriceRace(gaps, a.family, scales)
    acquired = json.loads((ROOT/ACQUIRED[a.family]).read_text()); assert acquired['status'] == 'completed'
    saved = torch.load(ROOT/acquired['final_weights'], weights_only=False); state = saved['model'] if 'model' in saved else saved
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith(('feat.', 'price.', 'log_scale')) for k in missing), (missing, unexpected)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(a.seed); s = dict(epoch=0, curve=[], best=None, wall_s=0., cpu_s=0.)
    if progress.exists():
        ck = torch.load(progress, weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state = ck['np_rng']; torch.set_rng_state(ck['torch_rng']); s = ck['state']
    else:
        start = evaluate(model, dev); s['curve'].append(dict(epoch=0, **start)); print(json.dumps(s['curve'][-1]), flush=True)
    segment = time.monotonic(); cpu = time.process_time(); wall0, cpu0 = s['wall_s'], s['cpu_s']
    while s['epoch'] < a.epochs:
        order = rng.permutation(len(train)); total = 0.; seen = 0
        for k, b in enumerate(range(0, len(order), 16)):
            if a.max_batches and k >= a.max_batches: break
            t, m, f, lo, hi = stack([train[i] for i in order[b:b+16]])
            tl, ml, pl, v = model.all_terms(t, m, f, lo, hi)
            loss = -((tl+ml).sum() + pl.sum())/tl.numel(); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss)*tl.numel(); seen += tl.numel()
        s['epoch'] += 1; s['wall_s'] = wall0+time.monotonic()-segment; s['cpu_s'] = cpu0+time.process_time()-cpu
        row = dict(epoch=s['epoch'], train_loss=total/seen, wall_s=s['wall_s'])
        if s['epoch'] % 2 == 0 or s['epoch'] == a.epochs:
            row.update(evaluate(model, dev))
            if s['best'] is None or row['return_nll_sum'] < s['best']['return_nll_sum']:
                s['best'] = dict(row); torch.save(dict(model=model.state_dict(), moments=moments, scales=scales, family=a.family), best_path)
        s['curve'].append(row); print(json.dumps(row), flush=True)
        tmp = progress.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), state=s), tmp); tmp.replace(progress)
    record = dict(status='completed', battle='B10', task='B10-P price', family=a.family, seed=a.seed, epochs=a.epochs, lr=a.lr, smoke=bool(a.max_batches),
                  initial_dev=s['curve'][0], best=s['best'], curve=s['curve'], final_weights=str(best_path.relative_to(ROOT)),
                  acquired_from=ACQUIRED[a.family], feature_moments=moments, label_scales=scales,
                  parameters=sum(x.numel() for x in model.parameters()), wall_s=s['wall_s'], cpu_s=s['cpu_s'],
                  process_wall_s=time.monotonic()-began, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={n: sha(ROOT/n) for n in SOURCES}, data_sha256=receipt(list(TRAIN)+list(VAL)), september_opened=False,
                  scope='Fixed-weight development comparison. Same 801 TRAIN/756 DEV windows as B10; DEV scored positions 128..1023, state reset per window. '
                        'All families receive identical causal inputs (marks, gaps, log size, tick move, past 60/300/1800 s realized variance, past-60 s trade count, 10 s signed-flow imbalance) and the same 3-component Gaussian-mixture return head per horizon, '
                        'scored as the exact 0.01 USDT tick cell of the realized return. Warm start from the acquired seed-0 tape model; new price parameters start prediction-neutral. Selection on DEV return likelihood; September sealed.')
    out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
