"""Curie chain v37 (replaces v36 while the hierarchy test runs; the v36 native step failed: the e2e smoke v2 stopped at 3.07 GB RSS
because per-correction clocks replicated each write list; fixed, contracts v25). Waits for the hierarchy test, then: native e2e smoke
v3 (--micro 2), coverage gate, RSS gate with microbatch fallback, the native member at 16M, the native member at 4M, then the
fixed CERL pair, PTB and the L2 diagnostic (v34/v33 steps)."""
import json, subprocess, time
from pathlib import Path
import continue_market_depth_v2 as market
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
from continue_curie_deep_recovery_v8 import completed
import continue_curie_deep_recovery_v34 as v34
import continue_curie_deep_recovery_v36 as v36

ROOT = Path(__file__).resolve().parents[1]
E2E = 'curie_r1_rmr_native_e2e_smoke_20261017_v3'; N4 = 'curie_r1_rmr_nativeev_K4_4M_s0_20261017_v1'


def main():
    while subprocess.run(['pgrep', '-f', f'^/workspace/.venv-docker/bin/python experiments/rhm_depth.py --tag {v34.RHM}'], capture_output=True).stdout: time.sleep(30)
    time.sleep(10)
    if (ROOT/f'experiments/results/diagnostics/{v34.RHM}.json').exists():
        market.publish([f'experiments/results/diagnostics/{v34.RHM}.json', 'experiments/results/diagnostics/curie_rhm_depth_contracts_20261017_v1.json'], 'R1 depth trainability on a synthetic hierarchy with an exact Bayes ladder (theory 173 section 14, theory 174)')
    rss, sfx, sps = float('inf'), '', None
    try:
        subprocess.run(['.venv-docker/bin/python', 'scripts/check_native_coverage.py'] + [f'experiments/queue/r1/{t}.txt' for t in (E2E, v36.GATE, v36.NATIVE, v36.NATIVE + '_m2', v36.NATIVE + '_m4', N4, N4 + '_m2', N4 + '_m4')], cwd=ROOT, check=True)
        if not completed(E2E): stable_host(); original.run(E2E, 3600)
        assert completed(E2E), 'native e2e smoke did not complete'
        for sfx in ('', '_m2', '_m4'):
            try: stable_host(); original.run(v36.GATE + sfx, 3600); rss, sps = v36.gate(v36.GATE + sfx)
            except Exception as error: rss, sps = float('inf'), None; print(json.dumps({v36.GATE + sfx: 'failed', 'error': repr(error)}), flush=True)
            print(json.dumps(dict(gate=v36.GATE + sfx, max_rss_kb=rss, s_per_step=sps)), flush=True)
            if rss < v36.RSS_MAX_KB: break
        if rss < v36.RSS_MAX_KB:
            if not completed(v36.NATIVE): stable_host(); original.run(v36.NATIVE + sfx, int(16384*sps*1.4 + 4*3600))
            if not completed(v36.NATIVE): raise RuntimeError('native run ended without a completed result')
            market.publish([f'experiments/results/token_language/{v36.NATIVE}.json', f'experiments/results/token_language/{E2E}.json'], 'R1 full native member at 16M (event spine, per-correction clocks, large concept bank, local clipping, online column)')
        else: print(json.dumps({v36.NATIVE: 'skipped', 'reason': f'gate RSS {rss} KB above {v36.RSS_MAX_KB} even with 4 microbatches'}), flush=True)
    except Exception as error: print(json.dumps({v36.NATIVE: 'failed', 'error': repr(error)}), flush=True)
    try:
        if rss < v36.RSS_MAX_KB and not completed(N4): stable_host(); original.run(N4 + sfx, int(4096*sps*1.4 + 3*3600))
        if completed(N4): market.publish([f'experiments/results/token_language/{N4}.json'], 'R1 full native member at 4M: the flagship slope point')
    except Exception as error: print(json.dumps({'native4M': 'failed', 'error': repr(error)}), flush=True)
    for tag, timeout in ((v34.CERL4, 4*3600), (v34.CERL16, 12*3600)):
        try:
            from continue_curie_deep_recovery_v8 import run_or_await
            run_or_await(tag, timeout); market.publish([f'experiments/results/token_language/{tag}.json'], f'R1 CERL with the capacity fixes ({tag.split("_")[4]}): usable capacity and the data slope')
        except Exception as error: print(json.dumps({tag: 'failed', 'error': repr(error)}), flush=True)
    v34.v33.main_after_sparse()


if __name__ == '__main__':
    main()
