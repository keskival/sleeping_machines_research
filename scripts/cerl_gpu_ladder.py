#!/usr/bin/env python3
"""Run the native CERL GPU ladder (experiments/gpu_ladder/r1_cerl_ladder_v1.json) on one GPU worker, one arm at a time through
run_safe.sh (immutable one-job queues with the admission declaration), smallest budgets first; then the analysis.
  --shard i/n splits arms across workers; SIGTERM stops at a checkpoint (exit 75 resumes exactly)."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def arms(spec):
    out = []
    for b in spec['budgets']:
        for arm in spec['arms']:
            tag = f"{spec['name']}_{arm.get('name', 'cerl')}_d{arm['d']}_u{arm['units']}_c{b:.0e}".replace('+', '') + '_s0'
            out.append(dict(tag=tag, budget=b, **arm))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', default=str(ROOT/'experiments/gpu_ladder/r1_cerl_ladder_v1.json')); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--out-dir', default=str(ROOT/'experiments/results/gpu_ladder')); a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text()); i, n = map(int, a.shard.split('/'))
    stop = {'f': False}; signal.signal(signal.SIGTERM, lambda *_: stop.update(f=True)); w = spec['worker']
    for k, arm in enumerate(arms(spec)):
        if k % n != i or stop['f']: continue
        res = Path(a.out_dir)/f"{arm['tag']}.json"
        if res.exists() and json.loads(res.read_text()).get('status') == 'completed': continue
        args = [f"--{key} {val}" for key, val in spec['common'].items()] + [f"--d {arm['d']}", f"--modes {arm['modes']}", f"--units {arm['units']}", f"--budget-flops {arm['budget']:g}", f"--out-dir {a.out_dir}"] + ([arm['args']] if arm.get('args') else [])
        qdir = ROOT/'experiments/queue/r1'/arm['tag']; qdir.mkdir(parents=True, exist_ok=True); q = qdir/f"{arm['tag']}.txt"
        if not q.exists():
            q.write_text(f"# Battle {spec['battle']}, native CERL GPU ladder {spec['name']} (budget {arm['budget']:.0e} FLOPs, d{arm['d']}, {arm['units']} concept units). Decision: {spec['decision']}\n"
                         + ''.join(f'# {d}\n' for d in spec['declaration']) + ' '.join([arm['tag'], spec['driver'], '--tag', arm['tag'], *args]) + '\n')
        env = dict(os.environ, MIN_AVAIL_MB=str(w['min_avail_mb']), MEM_CAP_RSS_KB=str(w['rss_cap_kb']), MEM_CAP_KB=str(w['vmem_cap_kb']), CGROUP_RESERVE_MB=str(w['cgroup_reserve_mb']),
                   JOB_TIMEOUT_S=str(int(w['timeout_s_per_1e16']*max(1.0, arm['budget']/1e16))), WAIT='1')
        print(json.dumps(dict(start=arm['tag'])), flush=True); p = subprocess.Popen(['bash', 'experiments/queue/run_safe.sh', str(q.relative_to(ROOT))], cwd=ROOT, env=env)
        while p.poll() is None:
            if stop['f']: p.send_signal(signal.SIGTERM)
            time.sleep(5)
        print(json.dumps(dict(end=arm['tag'], returncode=p.returncode)), flush=True)
        if p.returncode in (75, 143) or stop['f']: break
        subprocess.run([sys.executable, 'analysis/gpu_ladder_fit.py', '--name', spec['name'], '--out-dir', a.out_dir], cwd=ROOT)


if __name__ == '__main__':
    main()
