"""R1 part-whole hierarchy contracts on real tokens (no fitting): base equality at init, causality, exact span
integration against a sequential recurrence, gradient to every level parameter, observed boundary rates and smoke cost."""
import argparse, json, sys, time, hashlib, resource
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_part_whole_lm as pw
from r1_deep_keyed_lm import DeepKeyedLM
from r1_token_keyed_lm import load


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    tr = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); x = torch.from_numpy(np.asarray(tr[5_000_000:5_000_000 + 257]).astype(np.int64)).reshape(1, 257)
    out, checks = {}, {}
    # 1. At initialisation the hierarchy equals its plan-B base exactly.
    torch.manual_seed(0); base = DeepKeyedLM(64, 32, 2, 4, 16, 0.1); torch.manual_seed(0); model = pw.PartWholeLM(64, 32, 2, 4, 16, 0.1)
    missing = model.load_state_dict(base.state_dict(), strict=False); base.eval(); model.eval()
    with torch.no_grad():
        hb, eb = base.states(x[:, :-1]); hm, em = model.states(x[:, :-1]); rows = list(range(256))
        lb = base.logits(hb, eb, x[:, :-1], rows); lm = model.logits(hm, em, x[:, :-1], rows)
    checks['init_equals_base'] = bool(torch.equal(lb, lm)); out['unexpected_keys'] = list(missing.unexpected_keys)
    # 2. Causality with active levels.
    with torch.no_grad():
        for n, prm in model.named_parameters():
            if n.startswith(('level_in', 'mix.2')): prm.normal_(std=.05)
        x2 = x.clone(); x2[:, 100] = (x2[:, 100] + 777) % 50257
        l1 = model.logits(*model.states(x[:, :-1]), x[:, :-1], rows); l2 = model.logits(*model.states(x2[:, :-1]), x2[:, :-1], rows)
    out['prefix_max_change'] = float((l1[:, :100] - l2[:, :100]).abs().max()); checks['causal'] = out['prefix_max_change'] == 0 and bool((l1[:, 100:] - l2[:, 100:]).abs().max() > 0)
    # 3. Parallel span integration equals the sequential recurrence z_t = c z_{t-1} + w_t, reset at span starts (token level).
    lvl = model.levels[0]; s1 = pw.boundaries(x[:, :-1])[0]; u = torch.randn(1, 256, 64, dtype=torch.float64); lvl64 = lvl.double()
    t = torch.arange(256); span = s1.long().cumsum(1); mask = (t[:, None] >= t[None, :]) & (span[:, :, None] == span[:, None, :])
    with torch.no_grad():
        par = pw.masked_memory(lvl64.integ, u, (t[:, None] - t[None, :]).to(u.dtype).expand(1, -1, -1), mask)
        tm = lvl64.integ; n = tm.n; g = torch.sigmoid(tm.gate(u)); w = tm.write(u); wr, wi = w[..., :n]*g, w[..., n:]*g
        c = torch.exp(-tm.log_rate.exp()); cr, ci = c*torch.cos(tm.freq), c*torch.sin(tm.freq); zr = torch.zeros(1, n, dtype=u.dtype); zi = torch.zeros_like(zr); seq = []
        for k in range(256):
            keep = (~s1[:, k]).to(u.dtype)[:, None]; zr, zi = keep*(cr*zr - ci*zi) + wr[:, k], keep*(cr*zi + ci*zr) + wi[:, k]; seq.append(torch.cat([zr, zi], -1))
        seq = torch.stack(seq, 1)
    model.float(); out['integration_max_abs_error'] = float((par - seq).abs().max()); checks['integration_exact'] = out['integration_max_abs_error'] < 1e-9
    # 4. Gradient reaches every level parameter (next-token + level-matched future targets).
    torch.manual_seed(0); model = pw.PartWholeLM(64, 32, 2, 4, 16, 0.1); model.train()
    h, e = model.states(x[:, :-1]); loss = F.cross_entropy(model.logits(h, e, x[:, :-1], rows).reshape(-1, pw.V), x[:, 1:].reshape(-1)) + 0.1*model.future_loss(x)
    loss.backward(); dead = [n for n, prm in model.named_parameters() if n.startswith('levels') and (prm.grad is None or float(prm.grad.abs().max()) == 0)]
    out['level_parameters_without_gradient'] = dead; checks['level_credit'] = not dead
    # 6. Banded/event-compressed computation equals the full-kernel reference (float64, whole model, active levels).
    torch.manual_seed(1); m64 = pw.PartWholeLM(64, 32, 2, 4, 16, 0.1).double(); m64.eval()
    with torch.no_grad():
        for n, prm in m64.named_parameters():
            if n.startswith(('level_in', 'mix.2')): prm.normal_(std=.05)
        fast = m64.logits(*m64.states(x[:, :-1]), x[:, :-1], rows)
        for lv in m64.levels: lv.reference = True
        ref = m64.logits(*m64.states(x[:, :-1]), x[:, :-1], rows)
    out['fast_vs_reference_max_abs_logit_error'] = float((fast - ref).abs().max()); checks['fast_equals_reference'] = out['fast_vs_reference_max_abs_logit_error'] < 1e-9
    # 7. Same equality on a 1024-token lane: the word-level event scan crosses chunk boundaries (carried state).
    xl = torch.from_numpy(np.asarray(tr[7_000_000:7_000_000 + 1025]).astype(np.int64))[None]; rows_l = list(range(1024))
    with torch.no_grad():
        for lv in m64.levels: lv.reference = False
        fast_l = m64.logits(*m64.states(xl[:, :-1]), xl[:, :-1], rows_l); out['word_events_long_lane'] = int(pw.boundaries(xl[:, :-1])[0].sum())
        for lv in m64.levels: lv.reference = True
        ref_l = m64.logits(*m64.states(xl[:, :-1]), xl[:, :-1], rows_l)
    out['long_lane_fast_vs_reference_max_abs_logit_error'] = float((fast_l - ref_l).abs().max()); checks['long_lane_fast_equals_reference'] = out['long_lane_fast_vs_reference_max_abs_logit_error'] < 1e-9
    del m64, fast_l, ref_l
    # 5. Observed boundary rates on 1M training tokens.
    big = torch.from_numpy(np.asarray(tr[6_000_000:6_000_000 + 4096*256]).astype(np.int64)).reshape(4096, 256)
    out['span_start_rate'] = {name: float(s.double().mean()) for name, s in zip(('word', 'clause', 'sentence'), pw.boundaries(big))}
    out['mean_span_tokens'] = {k: 1/v for k, v in out['span_start_rate'].items()}
    rec = dict(status='completed' if all(checks.values()) else 'failed', battle='R1', checks=checks, measurements=out,
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/part_whole_contracts.py', 'experiments/r1_part_whole_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/tables/gpt2_token_roles.json')},
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began)
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(rec, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=rec['status'], checks=checks, **{k: out[k] for k in ('prefix_max_change', 'integration_max_abs_error', 'fast_vs_reference_max_abs_logit_error', 'long_lane_fast_vs_reference_max_abs_logit_error', 'word_events_long_lane', 'level_parameters_without_gradient', 'span_start_rate', 'mean_span_tokens')})))


if __name__ == '__main__':
    main()
