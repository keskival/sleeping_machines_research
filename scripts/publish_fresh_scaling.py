"""Publish completed fixed-prior fresh scaling; no pending cells."""
import argparse
import fcntl
import hashlib
import json
from continue_unique_language import ROOT, command


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('2m','8m'));a=p.parse_args()
    d=json.loads((ROOT/f'experiments/results/diagnostics/aws_fresh_scaling_{a.stage}_decision_20261012_v1.json').read_text())
    assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    text='Fixed-prior fresh-acquisition scaling completed. '
    for r in d['rows']:
        text+=(f"Pool{r['pool']}, {r['targets']}targets: final DEV{r['final_nll']:.6f}, selected DEV{r['selected_nll']:.6f}; "
            f"final-NLL improvement over previous acquisition{r['final_learning_gain']:+.6f}; complete training{r['training_s']:.3f}s, "
            f"{r['ms_per_target']:.4f}ms/target. Recorded acquisition prefix agrees exactly. ")
    text+=f"Selected pool{d['selected_pool']}. "+d['scope']
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS fresh-acquisition scaling — 12 October.** '),('report/I_SCIENCE.md','Current AWS fresh-acquisition scaling. ')]:
            path=ROOT/name
            parts=[s for s in path.read_text().split('\n\n') if not s.startswith(('**AWS fresh-acquisition scaling','Current AWS fresh-acquisition scaling'))]
            if name.startswith('experiments/'):path.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:path.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record completed fixed-prior acquisition scaling')
        command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish completed fresh-acquisition scaling')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
