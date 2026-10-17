"""Curie chain v35 (replaces v34, which failed every step at 14:10 UTC: its run_or_await looked for the decay job's run_safe
shell, which had died with the old tmux session, while the training process kept running and holding the host lock). Waits on
the training process itself, publishes the decay result, then runs v34's remaining steps unchanged."""
import json, subprocess, time
import continue_market_depth_v2 as market
from continue_curie_deep_recovery_v8 import completed
from continue_curie_deep_recovery_v31 import DECAY
import continue_curie_deep_recovery_v34 as v34


def running(tag):
    return bool(subprocess.run(['pgrep', '-f', f'^/workspace/.venv-docker/bin/python experiments/r1_rmr.py --tag {tag} '], capture_output=True).stdout)


def main():
    try:
        while running(DECAY): time.sleep(30)
        time.sleep(10)                                                                             # lock released with the process
        if not completed(DECAY): raise RuntimeError(f'{DECAY} ended without a completed result')
        market.publish([f'experiments/results/token_language/{DECAY}.json'], 'R1 native depth at 16M: 4-level residual memory with decaying old-slot refinement (long-lane repair)')
    except Exception as error: print(json.dumps({DECAY: 'failed', 'error': repr(error)}), flush=True)
    v34.after_decay()


if __name__ == '__main__':
    main()
