"""Bounded own-family depth/exposure pilot on the existing R1 token protocol."""
import ast
import json
import subprocess
import time
from continue_unique_language import ROOT, wait


def submit(tag, args, timeout, purpose):
    source = 'experiments/common_prior_token_language.py'
    deps = set()
    for node in ast.walk(ast.parse((ROOT/source).read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith('sleeping_machines/') and node.value.endswith('.py'):
                deps.add(node.value)
    cmd = [str(ROOT/'.venv-docker/bin/python'), 'experiments/tpp/submit.py', '--slot', '2',
           '--battle', 'R1', '--source', source, '--result-dir', 'experiments/results/token_language',
           '--timeout', str(timeout), '--rss-kb', '1200000', '--purpose', purpose, '--job', f'{tag}|{args}']
    for dep in sorted(deps): cmd += ['--dep', dep]
    print(subprocess.check_output(cmd, cwd=ROOT, text=True), flush=True)


def main():
    deadline = time.monotonic()+3*3600
    common = ('--exposure fresh --dev-tokens 2048 --chunk 64 --credit-window 16 --payload 24 '
              '--heads 2 --depth 4 --pool 4 --lanes 8 --eval-lanes 8 --readout-init frequency '
              '--input-init balanced --future-score-positions 4 --seed 6')
    smoke = 'aws_integrated_depth4_smoke_20261012_s6_v1'
    submit(smoke, common+' --train-tokens 512 --steps 2 --eval-every 1', 300,
           'R1: finite integrated depth4 token/route/credit smoke before registered exposure ladder; no architecture substitution')
    wait(smoke, deadline)
    for name, tokens, steps, every, cap in [('128k', 65536, 256, 64, 600), ('512k', 262144, 1024, 256, 1800)]:
        tag = f'aws_integrated_depth4_{name}_20261012_s6_v1'
        submit(tag, common+f' --train-tokens {tokens} --steps {steps} --eval-every {every}', cap,
               'R1: does depth4 earn its complete learning cost versus saved depth2 at two exposure levels? '
               'Predeclared one-seed ladder, no small-data quality veto, public/reserved4M untouched')
        wait(tag, deadline)
    cmd = [str(ROOT/'.venv-docker/bin/python'), 'experiments/tpp/submit.py', '--slot', '2', '--battle', 'R1',
           '--source', 'experiments/analyze_depth_language.py', '--result-dir', 'experiments/results/diagnostics',
           '--timeout', '60', '--rss-kb', '200000', '--purpose', 'R1: source-bound depth/exposure quality and full fitting-wall decision',
           '--job', 'aws_integrated_depth_exposure_20261012_v1|']
    for name in ('128k', '512k'):
        cmd += ['--requires-result', f'experiments/results/token_language/aws_integrated_depth4_{name}_20261012_s6_v1.json']
    print(subprocess.check_output(cmd, cwd=ROOT, text=True), flush=True)


if __name__ == '__main__': main()
