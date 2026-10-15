"""B10-P streaming fine-tune: continue a phase-1 price model on long contiguous TRAIN segments with numerical state carried
across 1024-trade chunks (gradient truncated at chunk boundaries, memory never reset inside a segment), so the learner is
trained in the regime it is deployed in. DEV selection by streaming evaluation on fixed contiguous DEV segments.

TRAIN: each Aug 1-24 day contributes SEGMENTS contiguous segments of SEGMENT trades at fixed, evenly spaced offsets.
DEV: each Aug 25-31 day contributes two fixed contiguous segments; positions after the first 1024 trades are scored."""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/market'))
from b10_depth_tpp_v1 import TRAIN, VAL
from b10p_data import split_windows, HORIZONS
from b10p_data_v2 import day_arrays
from b10p_models import PriceRace
from b10p_online_v2 import forecast, price_ll
from b10_online_stream import detach, revealed_terms

SEGMENT = 16384; SEGMENTS = 4; CHUNK = 1024; WARM = 1024
SOURCES = ['experiments/market/b10p_stream_fit.py', 'experiments/market/b10p_online_v2.py', 'experiments/market/b10p_online.py', 'experiments/market/b10p_models.py',
           'experiments/market/b10p_data.py', 'experiments/market/b10p_data_v2.py', 'experiments/market/b10_online_stream.py', 'experiments/market/b10_matched_models.py',
           'experiments/market/b10_depth_tpp_v1.py', 'experiments/market/b10_stable_clock_head.py', 'experiments/tpp/race_tpp_v5.py']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def segments(days, per, moments):
    out = []
    for day in days:
        a = day_arrays(day); n = len(a['t']); f = (a['feats'] - np.array(moments['mean']))/np.array(moments['std'])
        for k in range(per):
            s = int((n - SEGMENT - 1)*(k + 0.5)/per); sl = slice(s, s + SEGMENT + 1)
            out.append(dict(day=str(day), start=s, t=torch.tensor(a['t'][sl])[None], m=torch.tensor(a['mark'][sl])[None], f=torch.tensor(f[sl])[None],
                            lo=torch.tensor(a['lo'][sl])[None], hi=torch.tensor(a['hi'][sl])[None]))
        del a
    return out


def run_segment(model, seg, opt=None):
    """Chunks of CHUNK trades with carried state. Returns summed tape/return log-likelihood over scored positions (>= WARM)."""
    state = None; tape = 0.; tn = 0; ret = np.zeros(len(HORIZONS)); rn = np.zeros(len(HORIZONS))
    for c in range(0, SEGMENT, CHUNK):
        sl = slice(c, c + CHUNK); t, m, f = seg['t'][:, sl], seg['m'][:, sl], seg['f'][:, sl]; lo, hi = seg['lo'][:, sl], seg['hi'][:, sl]
        with torch.set_grad_enabled(opt is not None):
            params, head, state = forecast(model, t, m, f, state)
            tl = revealed_terms(model, params, t, seg['t'][:, c+1:c+CHUNK+1], seg['m'][:, c+1:c+CHUNK+1])
            valid = torch.isfinite(lo); pl = price_ll(model, head, torch.nan_to_num(lo, nan=0.), torch.nan_to_num(hi, nan=1.)); pl = torch.where(valid, pl, torch.zeros_like(pl))
            if opt is not None:
                loss = -(tl.sum() + pl.sum())/tl.numel(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        state = detach(state)
        if c >= WARM:
            tape -= float(tl.detach().sum()); tn += tl.numel(); ret -= pl.detach().sum((0, 1)).numpy(); rn += valid.sum((0, 1)).numpy()
    return tape, tn, ret, rn


def evaluate(model, dev):
    model.eval(); tape = 0.; tn = 0; ret = np.zeros(len(HORIZONS)); rn = np.zeros(len(HORIZONS))
    for seg in dev:
        a, b, c, d = run_segment(model, seg); tape += a; tn += b; ret += c; rn += d
    per = (ret/rn).tolist()
    return dict(tape_nll=tape/tn, return_nll=dict(zip([f'{int(h)}s' for h in HORIZONS], per)), return_nll_sum=float(sum(per)), return_targets=rn.astype(int).tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--fit', required=True); ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max-segments', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64); began = time.monotonic()
    out = ROOT/f'experiments/results/market/{a.tag}.json'; assert not out.exists(); progress = out.with_suffix('.progress.pt'); best_path = out.with_suffix('.best.pt')
    fit = json.loads((ROOT/a.fit).read_text()); assert fit['status'] == 'completed' and not fit['smoke']
    saved = torch.load(ROOT/fit['final_weights'], weights_only=False); moments = saved['moments']
    train_w = split_windows(TRAIN, 20); gaps = np.concatenate([np.diff(r['t']) for r in train_w]); gaps = gaps[gaps > 0]; del train_w
    torch.manual_seed(a.seed); model = PriceRace(gaps, fit['family'], saved['scales']); model.load_state_dict(saved['model'])
    train = segments(TRAIN, SEGMENTS, moments); dev = segments(VAL, 2, moments)
    if a.max_segments: train = train[:a.max_segments]; dev = dev[:2]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); rng = np.random.default_rng(a.seed); s = dict(epoch=0, curve=[], best=None, wall_s=0.)
    if progress.exists():
        ck = torch.load(progress, weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); rng.bit_generator.state = ck['np_rng']; s = ck['state']
    else:
        start = evaluate(model, dev); s['curve'].append(dict(epoch=0, **start)); s['best'] = dict(epoch=0, **start)
        torch.save(dict(model=model.state_dict(), moments=moments, scales=saved['scales'], family=fit['family']), best_path); print(json.dumps(s['curve'][-1]), flush=True)
    segment_clock = time.monotonic(); wall0 = s['wall_s']
    while s['epoch'] < a.epochs:
        model.train()
        for i in rng.permutation(len(train)): run_segment(model, train[i], opt)
        s['epoch'] += 1; s['wall_s'] = wall0 + time.monotonic() - segment_clock
        row = dict(epoch=s['epoch'], wall_s=s['wall_s'], **evaluate(model, dev))
        if row['return_nll_sum'] < s['best']['return_nll_sum']:
            s['best'] = dict(row); torch.save(dict(model=model.state_dict(), moments=moments, scales=saved['scales'], family=fit['family']), best_path)
        s['curve'].append(row); print(json.dumps(row), flush=True)
        tmp = progress.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), np_rng=rng.bit_generator.state, state=s), tmp); tmp.replace(progress)
    record = dict(status='completed', battle='B10', task='B10-P price streaming fine-tune', family=fit['family'], smoke=bool(a.max_segments), seed=a.seed, epochs=a.epochs, lr=a.lr,
                  parent=a.fit, initial_dev=s['curve'][0], best=s['best'], curve=s['curve'], final_weights=str(best_path.relative_to(ROOT)), feature_moments=moments,
                  label_scales=saved['scales'], wall_s=s['wall_s'], process_wall_s=time.monotonic() - began, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={n: sha(ROOT/n) for n in SOURCES}, input_sha256={a.fit: sha(ROOT/a.fit), fit['final_weights']: sha(ROOT/fit['final_weights'])}, september_opened=False,
                  scope=f'Phase-1 model continued on {SEGMENTS} contiguous {SEGMENT}-trade segments per TRAIN day with state carried across {CHUNK}-trade gradient chunks; '
                        f'DEV: two fixed contiguous segments per Aug 25-31 day, streaming state, positions after the first {WARM} trades scored. Look-back features include the previous day. Seed {a.seed}.')
    out.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
