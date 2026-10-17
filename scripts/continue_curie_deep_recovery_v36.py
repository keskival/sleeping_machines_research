"""Curie chain v36 (replaces v35 while it waits on the K4 decay job; founder: the flagship is the native member with every
advantage on, not a ReZero-like dense-read stack). After the decay job: the native e2e smoke with events, the RSS/time gate,
the full native member with the composing event spine at 16M, then the hierarchy test, the fixed CERL pair, PTB, L2."""
import json, re, time
from pathlib import Path
import continue_market_depth_v2 as market
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
from continue_curie_deep_recovery_v8 import run_or_await, completed
from continue_curie_deep_recovery_v31 import DECAY
import continue_curie_deep_recovery_v34 as v34
from continue_curie_deep_recovery_v35 import running

ROOT = Path(__file__).resolve().parents[1]
E2E, GATE, NATIVE = 'curie_r1_rmr_native_e2e_smoke_20261017_v2', 'curie_r1_rmr_nativeev_K4_16M_s0_20261017_v1_gate', 'curie_r1_rmr_nativeev_K4_16M_s0_20261017_v1'
RSS_MAX_KB = 2_400_000


def gate():
    m = re.findall(r'\{"smoke_steps".*\}', (ROOT/f'experiments/queue/logs/{GATE}.log').read_text()); assert m, 'gate printed no smoke line'
    g = json.loads(m[-1]); return g['max_rss_kb'], g['s_per_step']


def main():
    try:
        while running(DECAY): time.sleep(30)
        time.sleep(10)
        if not completed(DECAY): raise RuntimeError(f'{DECAY} ended without a completed result')
        market.publish([f'experiments/results/token_language/{DECAY}.json'], 'R1 native depth at 16M: 4-level residual memory with decaying old-slot refinement (long-lane repair)')
    except Exception as error: print(json.dumps({DECAY: 'failed', 'error': repr(error)}), flush=True)
    try:
        import subprocess                                                                          # native coverage gate: a flagship run with any native mechanism off is refused
        subprocess.run(['.venv-docker/bin/python', 'scripts/check_native_coverage.py'] + [f'experiments/queue/r1/{t}.txt' for t in (E2E, GATE, NATIVE)], cwd=ROOT, check=True)
        if not completed(E2E): stable_host(); original.run(E2E, 3600)
        assert completed(E2E), 'native e2e smoke did not complete'
        stable_host(); original.run(GATE, 3600); rss, sps = gate(); print(json.dumps(dict(gate=GATE, max_rss_kb=rss, s_per_step=sps)), flush=True)
        if rss < RSS_MAX_KB:
            run_or_await(NATIVE, int(16384*sps*1.4 + 4*3600))
            market.publish([f'experiments/results/token_language/{NATIVE}.json', f'experiments/results/token_language/{E2E}.json'], 'R1 full native member at 16M with the composing event spine (all native mechanisms)')
        else: print(json.dumps({NATIVE: 'skipped', 'reason': f'gate RSS {rss} KB above {RSS_MAX_KB}'}), flush=True)
    except Exception as error: print(json.dumps({NATIVE: 'failed', 'error': repr(error)}), flush=True)
    v34.after_decay()


if __name__ == '__main__':
    main()
