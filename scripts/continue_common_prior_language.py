"""Publish direct future-learning contrasts; gate unchanged confirmations."""
import ast
import json
from pathlib import Path
import subprocess
import time
from continue_unique_language import FOLDER, ROOT, publish, wait


def submit_confirmations():
    source='experiments/common_prior_token_language.py';deps=set()
    for n in ast.walk(ast.parse((ROOT/source).read_text())):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('sleeping_machines/') and n.value.endswith('.py'):deps.add(n.value)
    base=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','2','--battle','R1','--result-dir','experiments/results/token_language']
    cmd=base+['--source',source,'--timeout','600','--rss-kb','1200000','--purpose','R1: unchanged common-prior/common-acquired-state fresh versus replay confirmations after seed6 improves over both replay and stopping at the common branch']
    for dep in sorted(deps):cmd.extend(['--dep',dep])
    for seed in (7,8):
        for arm in ('repeated','fresh'):
            args=f'--exposure {arm} --train-tokens 65536 --dev-tokens 2048 --steps 256 --chunk 64 --credit-window 16 --payload 24 --heads 2 --depth 2 --pool 4 --lanes 8 --eval-lanes 8 --eval-every 64 --readout-init frequency --input-init balanced --future-score-positions 4 --seed {seed}'
            cmd.extend(['--job',f'aws_integrated_common_prior_{arm}_20261012_s{seed}_v1|{args}'])
    receipt=subprocess.check_output(cmd,cwd=ROOT,text=True)
    for seed in (7,8):
        cmd=base+['--source','experiments/analyze_common_prior_stream.py','--timeout','60','--rss-kb','200000','--purpose','R1: verify common acquired learner and direct future-learning gains before reporting confirmation',
                  '--job',f'aws_integrated_common_prior_decision_20261012_s{seed}_v1|--reference aws_integrated_common_prior_repeated_20261012_s{seed}_v1 --candidate aws_integrated_common_prior_fresh_20261012_s{seed}_v1']
        for arm in ('repeated','fresh'):
            cmd.extend(['--requires-result',f'experiments/results/token_language/aws_integrated_common_prior_{arm}_20261012_s{seed}_v1.json'])
        receipt+=subprocess.check_output(cmd,cwd=ROOT,text=True)
    return receipt



def main():
    deadline=time.monotonic()+3*3600; cohort=[]
    for seed in (6,7,8):
        d=wait(f'aws_integrated_common_prior_decision_20261012_s{seed}_v1',deadline);cohort.append(d)
        r,c=d['rows'];stop_gain=c['branch_dev_nll']-c['post_branch_dev_nll']
        text=f"Common-prior acquired-state intervention seed{seed}: after identical first128updates and exact full-state fingerprint, fresh second block versus replay yields post-branch DEV{c['post_branch_dev_nll']:.6f} versus{r['post_branch_dev_nll']:.6f}, gain{d['post_branch_quality_gain']:+.6f}. Fresh versus stopping at the shared branch: gain{stop_gain:+.6f}. Same65536token prior,131056total fitting targets/256updates; both reset forward state at the same boundary and retain actor/AdamW/RNG. Training{r['training_wall_s']:.3f}/{c['training_wall_s']:.3f}s. This is measured future-learning value of a data intervention, not a trained policy or incumbent comparison. Broader exposure-policy gain remains confirmed across three seeds (0.274752±0.038816NLL). "
        if seed==6:
            gate=d['development_gate_passed'] and d['selected_quality_gain']>=0 and stop_gain>0
            text+=('Unchanged seeds7/8 admitted only after these useful-quality gates.' if gate else 'Gate fails; no replication admitted. Next architectural decision follows actual diagnosis.')
            publish(text)
            if not gate:return
            print(submit_confirmations(),flush=True)
        else:
            text+=('Final unchanged confirmation pending.' if seed==7 else 'Three-seed common-state decisions complete. Next integrated natural-stream decision uses actual quality, work and future-learning contrasts; public validation and4M remain reserved.')
            publish(text)


if __name__=='__main__':main()
