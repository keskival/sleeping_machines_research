"""Publish each completed native-learning decision; current docs, bounded monitor."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]

def command(*args):subprocess.run(args,cwd=ROOT,check=True)

def main():
    deadline=time.monotonic()+4*3600
    for seed in [3,4,5]:
        path=ROOT/'experiments/results/credit'/f'aws_native_update_transfer_20261012_s{seed}_decision.json'
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('Native transfer publication waits for '+str(path))
            time.sleep(10)
        record=json.loads(path.read_text());assert record['status']=='completed'
        with open('/tmp/aws-language-publication.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
            if dirty:raise RuntimeError('Preserve tracked edits; publication deferred: '+dirty)
            command('git','pull','--rebase')
            details='; '.join(f"depth{r['depth']}: gain{r['final_quality_gain']:+.6f}, matched-work ratio{r['baseline_over_matched_work']:.3f}x" if r['baseline_over_matched_work'] else f"depth{r['depth']}: gain{r['final_quality_gain']:+.6f}, target not reached" for r in record['rows'])
            text=f'Current AWS acquired native update transfer completes seed{seed}: {details}. The frozen source-seed0 controller transfers to a fresh actor; static acquired group scales are the conditioning control. Source acquisition is charged separately; work ratios describe incremental reuse. All1400TaxiTRAIN sequences cropped32events,threepasses,128DEV sequences, complete fit/validation wall. Dense-write development stage; no publicTEST or architectural supremacy claim. Evidence aws_native_update_transfer_20261012_s{seed}_decision. '
            text+=('Unchanged-version replication is admitted automatically.' if record['continuation'] else 'Next: use completed quality/work and multiplier drift to choose the next integrated development change.')
            science=ROOT/'report/I_SCIENCE.md';s=science.read_text();paras=s.split('\n\n')
            paras=[p for p in paras if not p.startswith(('Current AWS acquired native update transfer','Current AWS acquired native update transfer completes seed'))]
            science.write_text('\n\n'.join(paras).rstrip()+'\n\n'+text+'\n')
            handoff=ROOT/'experiments/HANDOFF.md';s=handoff.read_text();paras=s.split('\n\n')
            paras=[p for p in paras if not p.startswith(('**AWS acquired native update transfer live','**AWS acquired native update transfer complete'))]
            handoff.write_text('**AWS acquired native update transfer complete.** '+text+'\n\n'+'\n\n'.join(paras).lstrip())
            command('git','add','report/I_SCIENCE.md','experiments/HANDOFF.md')
            command('git','commit','-m',f'Record native future-learning seed{seed} quality and complete work')
            command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
            command('git','add','REPORT.md','report')
            command('git','commit','-m',f'Publish current native-learning seed{seed} graphical record')
            command('git','pull','--rebase');command('git','push','origin','main')
        if not record['continuation']:break

if __name__=='__main__':main()
