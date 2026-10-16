#!/usr/bin/env python3
"""Plan and run the R1 GPU scaling ladder (experiments/runbooks/GPU_SCALING_LADDER.md) on one GPU worker.

  plan    measure train FLOPs/token, parameters and tokens/s for every (variant, shape) on this GPU; choose the arms of
          each compute budget (tokens/parameter inside the spec window, spread over model size), add the learning-rate
          sweep, depth probe and extra variants; write <name>_plan.json with the estimated GPU hours.
  run     train the planned arms one at a time (smallest budgets first, so diagnostics arrive early); skip completed arms,
          resume interrupted ones exactly; after the learning-rate sweep completes, set the ladder's LR multiplier from
          it; run the analysis after every arm. --shard i/n splits arms across workers; --max-hours stops admitting
          arms past an estimated GPU-hour cap. SIGTERM stops the current arm at a checkpoint (exit 75 = resumable).
  status  print progress.
One training process per GPU; never run two runners on one worker.
"""
import argparse, json, math, os, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; PY = sys.executable


def default_lr(d): return 3e-3*math.sqrt(256/d)


def tag_of(name, variant, d, L, budget, extra=''): return f'{name}_{variant}_d{d}_L{L}_c{budget:.0e}'.replace('+', '') + extra + '_s0'


def driver_args(spec, arm, lr_mult):
    c = dict(spec['common']); args = ['--variant', arm['variant'], '--d', str(arm['d']), '--layers', str(arm['layers']), '--train-tokens', str(arm['train_tokens'])]
    for k, v in c.items(): args += [f'--{k}', str(v)]
    args += ['--lr', f"{default_lr(arm['d'])*arm.get('lr_mult', lr_mult):.6g}"]
    return args


def measure(spec, variant, d, L, out_dir):
    common = ['--variant', variant, '--d', str(d), '--layers', str(L), '--lane', str(spec['common']['lane'])]
    p = json.loads(subprocess.run([PY, spec['driver'], '--mode', 'plan', *common], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    tag = f"timing_{spec['name']}_{variant}_d{d}_L{L}_{time.strftime('%Y%m%dT%H%M%S')}"; tdir = out_dir/'timing'; tdir.mkdir(parents=True, exist_ok=True)
    qdir = ROOT/'experiments/queue/r1'/tag; qdir.mkdir(parents=True, exist_ok=True); q = qdir/f'{tag}.txt'
    q.write_text(f"# Battle {spec['battle']}, GPU ladder {spec['name']} plan phase: 60-update timing smoke (tokens/s, MFU, GPU memory, RSS) that sizes the ladder arms. No result file is kept.\n"
                 + ' '.join([tag, spec['driver'], '--tag', tag, *common, '--batch', str(spec['common']['batch']), '--train-tokens', str(10**9), '--max-steps', '60', '--log-every', '20',
                             '--evals', '2', '--diags', '2', '--checkpoint-every', '1000000', '--out-dir', str(tdir)]) + '\n')
    w = spec['worker']; env = dict(os.environ, MIN_AVAIL_MB=str(w['min_avail_mb']), MEM_CAP_RSS_KB=str(w['rss_cap_kb']), MEM_CAP_KB=str(w['vmem_cap_kb']), CGROUP_RESERVE_MB=str(w['cgroup_reserve_mb']), JOB_TIMEOUT_S='3600', WAIT='1')
    subprocess.run(['bash', 'experiments/queue/run_safe.sh', str(q.relative_to(ROOT))], cwd=ROOT, env=env, capture_output=True)
    log = ROOT/f'experiments/queue/logs/{tag}.log'; lines = log.read_text().splitlines() if log.exists() else []
    logs = [json.loads(l) for l in lines if l.startswith('{"event": "log"')]
    p['tokens_per_s'] = logs[-1]['tokens_per_s'] if logs else None; p['mfu'] = logs[-1].get('mfu') if logs else None; p['gpu_mem_gb'] = logs[-1].get('gpu_mem_gb') if logs else None
    rss = [int(l.split('RSS=')[1].split('KB')[0]) for l in (qdir/f'runner_{tag}.out').read_text().splitlines() if 'RSS=' in l] if (qdir/f'runner_{tag}.out').exists() else []
    p['max_rss_kb_seen'] = max(rss) if rss else None
    for f in tdir.glob(f'{tag}*'): f.unlink()
    print(json.dumps(p), flush=True); return p


def plan(spec, out_dir):
    shapes = {(v, d, L) for v in spec['variants'] for d, L in spec['shapes']}
    shapes |= {(spec['lr_sweep']['variant'], d, L) for d, L in spec['lr_sweep']['shapes']}
    shapes |= {(v, spec['depth_probe']['d'], L) for v in spec['depth_probe']['variants'] for L in spec['depth_probe']['layers']}
    shapes |= {(v, d, L) for v in spec.get('extra_variants', {}) for d, L in spec['shapes']}
    m = {f'{v}_{d}_{L}': measure(spec, v, d, L, out_dir) for v, d, L in sorted(shapes)}
    arms, lo, hi = [], *spec['tokens_per_param']

    def arm(v, d, L, budget, kind, extra='', lr_mult=None):
        p = m[f'{v}_{d}_{L}']; tok = int(budget/p['train_flops_per_token']); a = dict(tag=tag_of(spec['name'], v, d, L, budget, extra), kind=kind, variant=v, d=d, layers=L, budget=budget,
                                                                           train_tokens=tok, train_flops_per_token=p['train_flops_per_token'], parameters=p['parameters'], tokens_per_param=tok/p['parameters'],
                                                                           est_hours=tok/p['tokens_per_s']/3600 if p['tokens_per_s'] else None)
        if lr_mult is not None: a['lr_mult'] = lr_mult
        return a

    def pick(v, budget):
        ok = [(d, L) for d, L in spec['shapes'] if lo <= budget/m[f'{v}_{d}_{L}']['train_flops_per_token']/m[f'{v}_{d}_{L}']['parameters'] <= hi]
        ok.sort(key=lambda s: m[f'{v}_{s[0]}_{s[1]}']['parameters']); k = spec['max_per_budget']
        return ok if len(ok) <= k else [ok[round(i*(len(ok) - 1)/(k - 1))] for i in range(k)]
    s = spec['lr_sweep']
    for d, L in s['shapes']:
        for mult in s['multipliers']: arms.append(arm(s['variant'], d, L, s['budget'], 'lr_sweep', f'_lr{mult:g}', mult))
    for budget in spec['budgets']:
        for v in spec['variants']: arms += [arm(v, d, L, budget, 'ladder') for d, L in pick(v, budget)]
        for v, e in spec.get('extra_variants', {}).items():
            if budget in e['budgets']: arms += [arm(v, d, L, budget, 'extra_variant') for d, L in pick(v, budget)]
        if budget == spec['depth_probe']['budget']:
            dp = spec['depth_probe']; arms += [arm(v, dp['d'], L, budget, 'depth_probe') for v in dp['variants'] for L in dp['layers']]
    seen, uniq = set(), []
    for a in arms:
        if a['tag'] not in seen: seen.add(a['tag']); uniq.append(a)
    res = dict(name=spec['name'], spec=spec, measurements=m, arms=uniq, est_gpu_hours=sum(a['est_hours'] or 0 for a in uniq), created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out_dir/f"{spec['name']}_plan.json").write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(dict(arms=len(uniq), est_gpu_hours=res['est_gpu_hours'])), flush=True)


def done(out_dir, tag):
    p = out_dir/f'{tag}.json'; return p.exists() and json.loads(p.read_text()).get('status') == 'completed'


def lr_multiplier(plan_, out_dir):
    """Geometric mean over swept widths of the multiplier with the lowest selection loss (1.0 until the sweep completes)."""
    sweep = [a for a in plan_['arms'] if a['kind'] == 'lr_sweep']
    if not sweep or not all(done(out_dir, a['tag']) for a in sweep): return 1.0, None
    best = {}
    for a in sweep:
        sel = json.loads((out_dir/f"{a['tag']}.json").read_text())['select_w1024']; key = (a['d'], a['layers'])
        if key not in best or sel < best[key][1]: best[key] = (a['lr_mult'], sel)
    return math.exp(sum(math.log(v[0]) for v in best.values())/len(best)), {f'd{k[0]}_L{k[1]}': v for k, v in best.items()}


def run(spec, out_dir, shard, max_hours, logs, kinds):
    """Each arm runs through experiments/queue/run_safe.sh as an immutable one-job queue (host lock, RSS watchdog, memory floor, timeout)."""
    plan_ = json.loads((out_dir/f"{spec['name']}_plan.json").read_text()); i, n = shard
    arms = [a for k, a in enumerate(sorted(plan_['arms'], key=lambda a: (a['kind'] != 'lr_sweep', a['budget'], a['tag']))) if k % n == i and (not kinds or a['kind'] in kinds)]
    used, stop = 0.0, {'flag': False}; signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    for a in arms:
        if done(out_dir, a['tag']): continue
        if stop['flag']: break
        if max_hours and used + (a['est_hours'] or 0) > max_hours: print(json.dumps(dict(stopped='max_hours', next=a['tag'])), flush=True); break
        mult, best = lr_multiplier(plan_, out_dir)
        if a['kind'] != 'lr_sweep' and best is None and any(x['kind'] == 'lr_sweep' for x in plan_['arms']):
            print(json.dumps(dict(waiting='lr_sweep incomplete; run the sweep shard first', arm=a['tag'])), flush=True); break
        qdir = ROOT/'experiments/queue/r1'/a['tag']; qdir.mkdir(parents=True, exist_ok=True); q = qdir/f"{a['tag']}.txt"
        line = ' '.join([a['tag'], spec['driver'], '--tag', a['tag'], *driver_args(spec, a, mult), '--out-dir', str(out_dir)])
        if not q.exists():                                              # immutable one-job queue with its admission declaration
            q.write_text(f"# Battle {spec['battle']}, GPU scaling ladder {spec['name']} ({a['kind']}, budget {a['budget']:.0e} FLOPs, {a['variant']} d{a['d']} L{a['layers']}). Decision: {spec['decision']}\n"
                         + ''.join(f'# {d}\n' for d in spec['declaration']) + line + '\n')
        w = spec['worker']; timeout = int((a['est_hours'] or 1)*3600*w['timeout_factor'] + w['timeout_extra_s'])
        env = dict(os.environ, MIN_AVAIL_MB=str(w['min_avail_mb']), MEM_CAP_RSS_KB=str(w['rss_cap_kb']), MEM_CAP_KB=str(w['vmem_cap_kb']), CGROUP_RESERVE_MB=str(w['cgroup_reserve_mb']),
                   JOB_TIMEOUT_S=str(timeout), WAIT='1', WAIT_TIMEOUT_S='600')
        print(json.dumps(dict(start=a['tag'], lr_mult=a.get('lr_mult', mult), est_hours=a['est_hours'], timeout_s=timeout, queue=str(q.relative_to(ROOT)))), flush=True); t0 = time.time()
        proc = subprocess.Popen(['bash', 'experiments/queue/run_safe.sh', str(q.relative_to(ROOT))], cwd=ROOT, env=env)
        while proc.poll() is None:
            if stop['flag']: proc.send_signal(signal.SIGTERM)
            time.sleep(5)
        used += (time.time() - t0)/3600; print(json.dumps(dict(end=a['tag'], returncode=proc.returncode, hours=round((time.time() - t0)/3600, 3), completed=done(out_dir, a['tag']))), flush=True)
        if proc.returncode in (75, 143) or stop['flag']: break
        subprocess.run([PY, 'analysis/gpu_ladder_fit.py', '--name', spec['name'], '--out-dir', str(out_dir)], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', default=str(ROOT/'experiments/gpu_ladder/r1_gpu_ladder_v1.json')); ap.add_argument('--phase', choices=('plan', 'run', 'status'), required=True)
    ap.add_argument('--shard', default='0/1'); ap.add_argument('--kinds', default='', help='comma list: lr_sweep, ladder, depth_probe, extra_variant'); ap.add_argument('--max-hours', type=float, default=0.0); ap.add_argument('--out-dir', default=str(ROOT/'experiments/results/gpu_ladder'))
    a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text()); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'experiments/queue/logs'; logs.mkdir(parents=True, exist_ok=True)
    if a.phase == 'plan': return plan(spec, out_dir)
    if a.phase == 'run': return run(spec, out_dir, tuple(int(x) for x in a.shard.split('/')), a.max_hours, logs, [k for k in a.kinds.split(',') if k])
    p = json.loads((out_dir/f"{spec['name']}_plan.json").read_text()); d = [x for x in p['arms'] if done(out_dir, x['tag'])]
    print(json.dumps(dict(arms=len(p['arms']), completed=len(d), est_gpu_hours=p['est_gpu_hours'], done_hours=sum(x['est_hours'] or 0 for x in d), lr=lr_multiplier(p, out_dir)), indent=1))


if __name__ == '__main__':
    main()
