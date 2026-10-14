"""Resume the preserved second checkpoint; charge observed interrupted work."""
import hashlib
import json
import math
import subprocess
from pathlib import Path
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host

ROOT = original.ROOT


def main():
    previous = 'curie_allocated_deep_d8_8m_20261013_s6_resume1_v1'
    receipt = json.loads((ROOT / f'experiments/results/token_language/{previous}_interruption.json').read_text())
    first = json.loads((ROOT / 'experiments/results/token_language/curie_allocated_deep_d8_8m_20261013_s6_v1_interruption.json').read_text())
    checkpoint = ROOT / f'experiments/results/token_language/{previous}.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt['checkpoint_sha256']
    rates = {d: original.load(f'curie_allocated_deep_d{d}_smoke_20261013_s6_v1')['train_wall_s_total'] / 32768 for d in (8, 16)}
    stages = ((8, '8m', 16384, receipt['recovery_run']),
              (16, '8m', 16384, 'curie_allocated_deep_d16_8m_20261013_s6_v1'),
              (8, '32m', 65536, 'curie_allocated_deep_d8_32m_20261013_s6_v1'))
    for depth, size, steps, tag in stages:
        stable_host()
        timeout = math.ceil(2 * rates[depth] * steps * 512 + 3600)
        if timeout > 604800:
            raise RuntimeError('Measured fit exceeds seven-day envelope')
        original.run(tag, max(timeout, 5400))
        packet = original.load(tag)
        if tag == receipt['recovery_run']:
            packet['resource_interruptions'] = [first, receipt]
            packet['whole_fit_process_wall_lower_bound_s'] = (first['runner_interval_s']
                + receipt['prior_runner_observed_interval_lower_bound_s'] + packet['process_wall_s'])
            packet['whole_fit_work_scope'] = 'Observed interrupted intervals plus resumed process wall; second interruption has an unknown tail. Lower bound, not exact total. Successful-trajectory training time excludes discarded work.'
            path = ROOT / f'experiments/results/token_language/{tag}.json'
            tmp = path.with_suffix('.work.tmp')
            tmp.write_text(json.dumps(packet, indent=2) + '\n')
            tmp.replace(path)
        quality = tag + '_quality'
        original.run(quality, 5400)
        original.load(quality, 'diagnostics')
        subprocess.run(['git', 'add', f'experiments/results/token_language/{tag}.json',
                        f'experiments/results/diagnostics/{quality}.json'], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', f'Complete R1 depth{depth} {size} acquisition and DEV diagnostics'], cwd=ROOT, check=True)
        print(f'Completed {tag}: final DEV {packet["curve"][-1]["dev_nll"]}', flush=True)


if __name__ == '__main__':
    main()
