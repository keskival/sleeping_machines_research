"""Publish immutable aggregate snapshots and the complete official result."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from experiments.public_benchmarks.collect_neurobench_mg import collect
import publish_aws_language_progress as P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    status = Path(args.manifest).parent/'neurobench_summary.status.json'
    state = dict(status='monitoring', published=[])
    while True:
        result = collect()
        count = result['completed_repeats']
        if count and count not in state['published']:
            out = Path(f'experiments/results/diagnostics/aws_mg_official_r1_summary_n{count}_20261004T184500Z.json')
            if out.exists():
                assert P.git('ls-files', '--', str(out)), 'Preserved unpublished snapshot requires recovery'
            else:
                out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                files = [str(out)]
                if count == 30:
                    appendix = Path('report/appendices/aws_mg_official_r1_20261004.md')
                    assert not appendix.exists()
                    appendix.write_text('# Completed official Mackey–Glass r1\n\n'
                        f"All30 prescribed repeats completed. Fixed primary mix8 mean sMAPE: **{result['mean_smape_by_mode']['mix8']:.6f}**.\n\n"
                        'Argmax and sampled are reporting-only variants. Overlapping repeats do not supply an independent-sample confidence interval.\n\n'
                        'This is a quality result; inference FLOPs, repeated-stream state traffic and energy remain unmeasured. '
                        'The current driver does not retain per-repeat final weights, so trained serving parity is a separate future protocol.\n\n'
                        f'[Source-bound aggregate](../../{out})\n')
                    files.append(str(appendix))
                with open('/tmp/aws-language-publication.lock', 'a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    pid = P.coordinator(args.manifest)
                    os.kill(pid, signal.SIGSTOP)
                    try:
                        for _ in range(30):
                            if not P.git_children(pid):
                                break
                            time.sleep(1)
                        else:
                            raise RuntimeError('Coordinator Git children active')
                        P.git('add', '--', *files)
                        P.git('commit', '--only', '-m', f'Publish fixed NeuroBench MG aggregate {count}/30 repeats', '--', *files)
                        for _ in range(6):
                            P.git('pull', '--rebase')
                            try:
                                P.git('push', 'origin', 'main')
                                break
                            except Exception:
                                time.sleep(2)
                        else:
                            raise RuntimeError('Push retries exhausted')
                    finally:
                        os.kill(pid, signal.SIGCONT)
            state['published'].append(count)
        state.update(completed_repeats=count)
        if count == 30:
            state['status'] = 'completed'
        status.write_text(json.dumps(state, indent=2)+'\n')
        if count == 30:
            return
        lifecycle = json.loads(Path(args.manifest).with_name('worker_recovery.status.json').read_text())
        if lifecycle['status'] in ('needs_review', 'completed'):
            state.update(status='stopped_without_complete_protocol')
            status.write_text(json.dumps(state, indent=2)+'\n')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
