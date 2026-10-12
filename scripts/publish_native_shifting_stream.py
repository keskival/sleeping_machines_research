"""Publish each completed native-learning decision; current docs, bounded monitor."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]

def command(*args):subprocess.run(args,cwd=ROOT,check=True)

def main():
    deadline=time.monotonic()+12*3600
    for seed in [9,10,11]:
        path=ROOT/'experiments/results/credit'/f'aws_native_shifting_stream_20261012_s{seed}_decision.json'
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('Native transfer publication waits for '+str(path))
            time.sleep(10)
        record=json.loads(path.read_text());assert record['status']=='completed'
        with open('/tmp/aws-language-publication.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
            if dirty:raise RuntimeError('Preserve tracked edits; publication deferred: '+dirty)
            command('git','pull','--rebase')
            details="; ".join(f"depth{r['depth']}: cumulative online NLL{r['online_cumulative_nll']:.6f}, gain vs frozen{next(c['cumulative_quality_gain'] for c in r['comparisons'] if c['reference']=='frozen_acquired_policy'):+.6f}" for r in record['rows'])
            text=f'Current AWS native changing-stream learning completes seed{seed}: {details}. Score before feedback on identical unique targets. Online acquired policy compared with frozen/static/fixedAdam; source acquisition separately charged and exact actor backprop retained. Onepass over12800distinct RetweetTRAIN sequences cropped32events/396800targets, four constructed timing/mark-dependency regimes; complete measured fitting wall. No natural-chronology claim. Dense-write development stage; no publicTEST or architectural supremacy claim. Evidence aws_native_shifting_stream_20261012_s{seed}_decision. '
            text+=('Unchanged-version replication is admitted automatically.' if record['continuation'] else 'Next: use completed quality/work and multiplier drift to choose the next integrated development change.')
            science=ROOT/'report/I_SCIENCE.md';s=science.read_text();paras=s.split('\n\n')
            paras=[p for p in paras if not p.startswith(('Current AWS native changing-stream learning','Current AWS native changing-stream learning completes seed'))]
            science.write_text('\n\n'.join(paras).rstrip()+'\n\n'+text+'\n')
            handoff=ROOT/'experiments/HANDOFF.md';s=handoff.read_text();paras=s.split('\n\n')
            paras=[p for p in paras if not p.startswith(('**AWS native changing-stream learning live','**AWS native changing-stream learning complete'))]
            handoff.write_text('**AWS native changing-stream learning complete.** '+text+'\n\n'+'\n\n'.join(paras).lstrip())
            command('git','add','report/I_SCIENCE.md','experiments/HANDOFF.md')
            command('git','commit','-m',f'Record native future-learning seed{seed} quality and complete work')
            command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
            command('git','add','REPORT.md','report')
            command('git','commit','-m',f'Publish current native-learning seed{seed} graphical record')
            command('git','pull','--rebase');command('git','push','origin','main')
        if not record['continuation']:break

if __name__=='__main__':main()
