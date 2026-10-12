"""Bounded R1 publication and quality-gated larger distinct-experience pair.

Reads completed decisions; every new fit/evaluation is submitted to run_safe.
"""
import ast
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/results/token_language'


def command(*args):
    return subprocess.check_output(args,cwd=ROOT,text=True)


def wait(tag,deadline):
    path=FOLDER/f'{tag}.json'
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(tag)
        time.sleep(10)
    d=json.loads(path.read_text());assert d['status']=='completed';return d


def publish(text):
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():
            raise RuntimeError('Preserve tracked edits; publication deferred')
        command('git','pull','--rebase')
        for name in ('experiments/HANDOFF.md','report/I_SCIENCE.md'):
            p=ROOT/name;parts=p.read_text().split('\n\n')
            parts=[v for v in parts if not v.startswith(('**AWS distinct-experience','Current AWS distinct-experience'))]
            if name.startswith('experiments/'):
                p.write_text('**AWS distinct-experience language development.** '+text+'\n\n'+'\n\n'.join(parts))
            else:p.write_text('\n\n'.join(parts).rstrip()+'\n\n'+'Current AWS distinct-experience language development. '+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md')
        command('git','commit','-m','Record current distinct-experience integrated language decision')
        command('.venv-docker/bin/python','report/make_pdf.py')
        command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        # Rebuilds can be identical if no additional report packet has arrived.
        if command('git','diff','--cached','--name-only').strip():
            command('git','commit','-m','Publish current distinct-experience language evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


def submit_larger():
    deps={'experiments/horizon_token_language_engine.py','experiments/batched_token_episodes.py'}
    for n in ast.walk(ast.parse((ROOT/'experiments/horizon_token_language_engine.py').read_text())):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('sleeping_machines/') and n.value.endswith('.py'):deps.add(n.value)
    cmd=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','2','--battle','R1','--source','experiments/profile_integrated_language_v2.py','--result-dir','experiments/results/token_language','--timeout','1800','--rss-kb','1200000','--purpose','R1: larger distinct-experience confirmation after three-seed same-target gain; same524272targets/1024updates, two passes262144 versus one pass524280; retain P24 mechanisms and sealed public/reserved4M data']
    for dep in sorted(deps):cmd.extend(['--dep',dep])
    for arm,n in [('repeated',262144),('unique',524280)]:
        args=f'--train-tokens {n} --dev-tokens 2048 --steps 1024 --chunk 64 --credit-window 16 --payload 24 --heads 2 --depth 2 --pool 4 --lanes 8 --eval-lanes 8 --eval-every 256 --readout-init frequency --input-init balanced --future-score-positions 4 --seed 6'
        cmd.extend(['--job',f'aws_integrated_p24_{arm}_512k_budget_20261012_s6_v1|{args}'])
    receipt=command(*cmd)
    cmd=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','2','--battle','R1','--source','experiments/analyze_unique_token_stream_v2.py','--result-dir','experiments/results/token_language','--timeout','180','--rss-kb','1200000','--purpose','R1: larger fixed-target quality/work/context decision; actual larger evidence only, no extrapolated score','--job','aws_integrated_p24_unique_512k_decision_20261012_s6_v1|--reference aws_integrated_p24_repeated_512k_budget_20261012_s6_v1 --candidate aws_integrated_p24_unique_512k_budget_20261012_s6_v1']
    for dep in sorted((deps-{'experiments/batched_token_episodes.py'})|{'experiments/profile_integrated_language_v2.py'}):cmd.extend(['--dep',dep])
    return receipt+command(*cmd)


def main():
    deadline=time.monotonic()+3*3600
    cohort=[]
    for seed in (6,7,8):
        d=wait(f'aws_integrated_p24_unique_stream_decision_20261012_s{seed}_v1',deadline);cohort.append(d)
        r,c=d['rows']
        text=f"Completed seed{seed}: repeated DEV{r['selected_dev_nll']:.6f}, distinct DEV{c['selected_dev_nll']:.6f}, gain{d['quality_gain']:+.6f}; training{r['training_wall_s']:.3f}/{c['training_wall_s']:.3f}s. Same131056fitting targets/256updates;65536two-pass versus131064one-pass GPT2 FineWeb tokens. TRAIN frequency-prior gain{d['initial_prior_gain']:+.6f}; post-initialization learning gain change{d['learning_gain_change']:+.6f}; contextual gain change{d['context_gain_change']:+.6f}. Integrated P24D2H2U4/credit16 retained; natural data, no nonstationarity or Transformer claim. "
        text+=('Unchanged seed confirmations running.' if seed!=8 else 'Three-seed development decisions complete; larger stage gated on all positive quality gains, positive mean learning-gain change and at most5% mean training-wall increase.')
        publish(text)
    ratios=[d['rows'][0]['training_wall_s']/d['rows'][1]['training_wall_s'] for d in cohort]
    gate=all(d['quality_gain']>0 and d['rows'][1]['context_gain']>0 for d in cohort) and statistics.mean(d['learning_gain_change'] for d in cohort)>0 and statistics.mean(ratios)>=1/1.05
    if not gate:
        print('STOP: larger-stage quality/work gate failed',flush=True);return
    print(submit_larger(),flush=True)
    d=wait('aws_integrated_p24_unique_512k_decision_20261012_s6_v1',deadline)
    r,c=d['rows']
    publish(f"Larger same-target stage completed: repeated DEV{r['selected_dev_nll']:.6f}, distinct DEV{c['selected_dev_nll']:.6f}, gain{d['quality_gain']:+.6f}; training{r['training_wall_s']:.3f}/{c['training_wall_s']:.3f}s. Same524272fitting targets/1024updates;262144two-pass versus524280one-pass tokens. Initial-prior gain{d['initial_prior_gain']:+.6f}; learning-gain change{d['learning_gain_change']:+.6f}; context-gain change{d['context_gain_change']:+.6f}. Single-seed larger development point; three-seed smaller decisions preserved. Original integrated model retained, public validation and4M reserved. Next decision is mechanism-specific capacity and learning allocation on natural streams using completed quality/work, with Curie compact-credit ownership preserved.")


if __name__=='__main__':main()
