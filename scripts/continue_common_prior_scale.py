"""One larger common-state point, gated on three useful small confirmations."""
import ast
import statistics
import subprocess
import time
from continue_unique_language import ROOT, publish, wait


def main():
    deadline=time.monotonic()+3*3600
    cohort=[wait(f'aws_integrated_common_prior_decision_20261012_s{s}_v1',deadline) for s in (6,7,8)]
    gains=[d['rows'][1]['branch_dev_nll']-d['rows'][1]['post_branch_dev_nll'] for d in cohort]
    gate=all(d['development_gate_passed'] and d['selected_quality_gain']>=0 for d in cohort) and all(g>0 for g in gains)
    if not gate:
        print('STOP: common-state quality/work/stop gates failed',flush=True);return
    source='experiments/common_prior_token_language.py';deps=set()
    for n in ast.walk(ast.parse((ROOT/source).read_text())):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('sleeping_machines/') and n.value.endswith('.py'):deps.add(n.value)
    base=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','2','--battle','R1','--result-dir','experiments/results/token_language']
    cmd=base+['--source',source,'--timeout','1800','--rss-kb','1200000','--purpose','R1: one larger common-acquired-state future-learning contrast after three seeds beat replay and stopping with equal prior;524272targets/1024updates, public and4M protected']
    for dep in sorted(deps):cmd.extend(['--dep',dep])
    for arm in ('repeated','fresh'):
        args=f'--exposure {arm} --train-tokens 262144 --dev-tokens 2048 --steps 1024 --chunk 64 --credit-window 16 --payload 24 --heads 2 --depth 2 --pool 4 --lanes 8 --eval-lanes 8 --eval-every 256 --readout-init frequency --input-init balanced --future-score-positions 4 --seed 6'
        cmd.extend(['--job',f'aws_integrated_common_prior_512k_{arm}_20261012_s6_v1|{args}'])
    print(subprocess.check_output(cmd,cwd=ROOT,text=True),flush=True)
    cmd=base+['--source','experiments/analyze_common_prior_stream_v2.py','--timeout','60','--rss-kb','200000','--purpose','R1: larger common-prior future-learning effect after exact first512-update full-state identity',
        '--job','aws_integrated_common_prior_512k_decision_20261012_s6_v1|--reference aws_integrated_common_prior_512k_repeated_20261012_s6_v1 --candidate aws_integrated_common_prior_512k_fresh_20261012_s6_v1']
    for arm in ('repeated','fresh'):
        cmd.extend(['--requires-result',f'experiments/results/token_language/aws_integrated_common_prior_512k_{arm}_20261012_s6_v1.json'])
    print(subprocess.check_output(cmd,cwd=ROOT,text=True),flush=True)
    d=wait('aws_integrated_common_prior_512k_decision_20261012_s6_v1',deadline)
    r,c=d['rows'];stop_gain=c['branch_dev_nll']-c['post_branch_dev_nll']
    publish(f"Larger common-acquired-state intervention complete: exact full-state fingerprint after512updates, same262144token prior; next512updates repeat or refresh that block. Fresh post-branch DEV{c['post_branch_dev_nll']:.6f} versus replay{r['post_branch_dev_nll']:.6f}, gain{d['post_branch_quality_gain']:+.6f}; fresh versus stopping gain{stop_gain:+.6f}. Training{r['training_wall_s']:.3f}/{c['training_wall_s']:.3f}s for the same524272targets/1024updates. Smaller common-state fresh-versus-stop gains mean{statistics.mean(gains):.6f}±{statistics.stdev(gains):.6f} across three seeds. One larger development seed. Core temporal/sparse/counterfactual mechanisms retained; policy-estimator training and useful-capacity scaling are the next integration decisions. No official or reserved4M scoring.")


if __name__=='__main__':main()
