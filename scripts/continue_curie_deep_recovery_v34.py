"""Curie chain v34 (supersedes v33 while the K4 decay run is running; awaited unchanged). Theory 173 section 14 (founder: depth is
trainable composition plus depth that grows with log data): after the decay run, the synthetic-hierarchy depth test, then the
CERL capacity pair (4M and 16M with the capacity fixes: does usable capacity steepen the slope?), then the sparse native member
(e2e smoke, RSS/time gate, 16M run), then the PTB fit resume + validation scoring, then the dense 16M L2 diagnostic."""
import json
from pathlib import Path
import continue_market_depth_v2 as market
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
from continue_curie_deep_recovery_v8 import run_or_await
from continue_curie_deep_recovery_v31 import DECAY
import continue_curie_deep_recovery_v33 as v33

ROOT = Path(__file__).resolve().parents[1]
RHM = 'curie_rhm_depth_K124_20261017_v1'
CERL4, CERL16 = 'curie_r1_cerl_fix_4M_s0_20261017_v1', 'curie_r1_cerl_fix_16M_s0_20261017_v1'


def main():
    try:
        run_or_await(DECAY, 64800); market.publish([f'experiments/results/token_language/{DECAY}.json'], 'R1 native depth at 16M: 4-level residual memory with decaying old-slot refinement (long-lane repair)')
    except Exception as error: print(json.dumps({DECAY: 'failed', 'error': repr(error)}), flush=True)
    after_decay()


def after_decay():
    try:
        out = ROOT/f'experiments/results/diagnostics/{RHM}.json'
        if not out.exists(): stable_host(); original.run(RHM, 6*3600)
        assert out.exists(), 'hierarchy depth test ended without a result'
        market.publish([f'experiments/results/diagnostics/{RHM}.json', 'experiments/results/diagnostics/curie_rhm_depth_contracts_20261017_v1.json'], 'R1 depth trainability on a synthetic hierarchy with an exact Bayes ladder (theory 173 section 14)')
    except Exception as error: print(json.dumps({RHM: 'failed', 'error': repr(error)}), flush=True)
    for tag, timeout in ((CERL4, 4*3600), (CERL16, 12*3600)):
        try:
            run_or_await(tag, timeout); market.publish([f'experiments/results/token_language/{tag}.json'], f'R1 CERL with the capacity fixes ({tag.split("_")[4]}): usable capacity and the data slope')
        except Exception as error: print(json.dumps({tag: 'failed', 'error': repr(error)}), flush=True)
    v33.main_after_sparse()


if __name__ == '__main__':
    main()
