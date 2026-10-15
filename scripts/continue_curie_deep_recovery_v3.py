"""Resume D16/8M after the 15 October container restart, then the admitted B10 depth stage."""
import hashlib
import json
import math
import subprocess
import continue_curie_deep_language as original
import continue_market_depth_v2 as market
from continue_curie_deep_recovery import stable_host

ROOT = original.ROOT
OLD = 'curie_allocated_deep_d16_8m_20261013_s6_v1'
NEW = 'curie_allocated_deep_d16_8m_20261013_s6_resume1_v1'


def main():
    receipt = json.loads((ROOT / f'experiments/results/token_language/{OLD}_interruption.json').read_text())
    checkpoint = ROOT / f'experiments/results/token_language/{OLD}.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt['checkpoint_sha256']
    rate = original.load('curie_allocated_deep_d16_smoke_20261013_s6_v1')['train_wall_s_total'] / 32768
    remaining = 16384 - receipt['checkpoint_step']
    stable_host()
    original.run(NEW, max(math.ceil(2 * rate * remaining * 512 + 3600), 5400))
    packet = original.load(NEW)
    packet['resource_interruptions'] = [receipt]
    packet['whole_fit_process_wall_lower_bound_s'] = receipt['prior_runner_observed_interval_lower_bound_s'] + packet['process_wall_s']
    packet['whole_fit_work_scope'] = 'Observed interrupted runner interval plus resumed process wall. Lower bound; successful-trajectory training time excludes discarded work after the step-15360 checkpoint.'
    path = ROOT / f'experiments/results/token_language/{NEW}.json'
    tmp = path.with_suffix('.work.tmp')
    tmp.write_text(json.dumps(packet, indent=2) + '\n')
    tmp.replace(path)
    original.run(NEW + '_quality', 5400)
    original.load(NEW + '_quality', 'diagnostics')
    market.publish([f'experiments/results/token_language/{NEW}.json', f'experiments/results/diagnostics/{NEW}_quality.json',
                    f'experiments/results/token_language/{OLD}_interruption.json'],
                   'Complete R1 depth16 8M acquisition (exact resume) and DEV diagnostics')
    print(f'Completed {NEW}: final DEV {packet["curve"][-1]["dev_nll"]}', flush=True)
    try:
        market.market_stage()
    except Exception as error:
        print(json.dumps({'market_stage': 'failed', 'error': repr(error), 'next': 'stop; inspect completed repair evidence'}), flush=True)
        failure = ROOT / f'experiments/results/market/{market.FIT}_failure.json'
        with failure.open('x') as f:
            json.dump(dict(status='failed', battle='B10', error=repr(error), smoke=market.SMOKE, fit=market.FIT), f, indent=2)
            f.write('\n')
    print('Queue complete: no unchanged 32M or duplicate horizon fit admitted', flush=True)


if __name__ == '__main__':
    main()
