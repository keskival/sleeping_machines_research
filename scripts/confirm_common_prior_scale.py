"""Bounded R1 larger-exposure confirmation; immutable existing fit recipe."""
import ast
import statistics
import subprocess
import time

from continue_unique_language import ROOT, publish, wait


def main():
    deadline = time.monotonic() + 3 * 3600
    first = wait('aws_integrated_common_prior_512k_decision_20261012_s6_v1', deadline)
    assert first['development_gate_passed'] and first['selected_quality_gain'] > 0
    source = 'experiments/common_prior_token_language.py'
    deps = set()
    for node in ast.walk(ast.parse((ROOT / source).read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith('sleeping_machines/') and node.value.endswith('.py'):
                deps.add(node.value)
    base = [str(ROOT / '.venv-docker/bin/python'), 'experiments/tpp/submit.py',
            '--slot', '2', '--battle', 'R1', '--result-dir', 'experiments/results/token_language']
    for seed in (7, 8):
        cmd = base + ['--source', source, '--timeout', '1800', '--rss-kb', '1200000',
                      '--purpose', 'R1: confirm larger common-state future-learning return across seeds; decide whether to advance intervention-policy acquisition, same 524272 targets, sealed public/4M untouched']
        for dep in sorted(deps):
            cmd.extend(['--dep', dep])
        for arm in ('repeated', 'fresh'):
            args = (f'--exposure {arm} --train-tokens 262144 --dev-tokens 2048 --steps 1024 '
                    '--chunk 64 --credit-window 16 --payload 24 --heads 2 --depth 2 --pool 4 '
                    '--lanes 8 --eval-lanes 8 --eval-every 256 --readout-init frequency '
                    f'--input-init balanced --future-score-positions 4 --seed {seed}')
            cmd.extend(['--job', f'aws_integrated_common_prior_512k_{arm}_20261012_s{seed}_v1|{args}'])
        print(subprocess.check_output(cmd, cwd=ROOT, text=True), flush=True)
        reference = f'aws_integrated_common_prior_512k_repeated_20261012_s{seed}_v1'
        candidate = f'aws_integrated_common_prior_512k_fresh_20261012_s{seed}_v1'
        cmd = base + ['--source', 'experiments/analyze_common_prior_stream_v2.py',
                      '--timeout', '60', '--rss-kb', '200000',
                      '--purpose', 'R1: verify exact common learner and larger future-quality/work confirmation',
                      '--job', f'aws_integrated_common_prior_512k_decision_20261012_s{seed}_v1|--reference {reference} --candidate {candidate}']
        for tag in (reference, candidate):
            cmd.extend(['--requires-result', f'experiments/results/token_language/{tag}.json'])
        print(subprocess.check_output(cmd, cwd=ROOT, text=True), flush=True)
    records = [first] + [wait(f'aws_integrated_common_prior_512k_decision_20261012_s{s}_v1', deadline) for s in (7, 8)]
    gains = [d['post_branch_quality_gain'] for d in records]
    stops = [d['rows'][1]['branch_dev_nll'] - d['rows'][1]['post_branch_dev_nll'] for d in records]
    gate = all(d['development_gate_passed'] and d['selected_quality_gain'] > 0 for d in records)
    gate = gate and all(g > 0 for g in stops)
    publish(f"Larger common-state cohort complete, seeds 6/7/8: fresh versus best post-branch replay gain "
            f"{statistics.mean(gains):.6f} ± {statistics.stdev(gains):.6f} NLL; fresh versus shared-branch stopping gain "
            f"{statistics.mean(stops):.6f} ± {statistics.stdev(stops):.6f} (sample SD). "
            f"Quality/work gate {'PASS' if gate else 'FAIL'}. Each pair shares its first 262144-token prior and exact "
            "512-update learner fingerprint; total 524272 fitting targets / 1024 updates. "
            "Replay/fresh have equal training targets; stopping uses fewer updates. All completed per-seed results are preserved. "
            "Integrated P24D2H2U4/credit16 retained. Next intervention-policy acquisition must include changing regimes "
            "and competing useful actions: these homogeneous contrasts do not train when to replay. Public validation "
            "and reserved 4M remain untouched; Curie owns compact/deep credit. Within-family development, no fitted scaling exponent.")


if __name__ == '__main__':
    main()
