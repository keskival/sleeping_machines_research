"""Publish completed capacity development with its measured verdict."""
import fcntl
import hashlib
import json
from continue_unique_language import ROOT, command


def main():
    packet=ROOT/'experiments/results/diagnostics/aws_capacity_acquisition_20261012_v1.json'
    d=json.loads(packet.read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    text='First-pass addressed-capacity development completes at two acquisition sizes. '
    for level in d['levels']:
        r,c=level['rows']
        text+=(f"At{c['targets']}targets, pool4/8 selected DEV{r['selected_nll']:.6f}/{c['selected_nll']:.6f}; "
            f"complete training{r['complete_training_s']:.3f}/{c['complete_training_s']:.3f}s. ")
    text+='Selected writes stay4/target, scored keys16→32 and available receivers16→32. '+d['scope']
    text+=' Use quality/work and receiver specialization to select the next member; no first-pass family verdict.'
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS addressed-capacity acquisition — 12 October.** '),
                            ('report/I_SCIENCE.md','Current AWS addressed-capacity acquisition. ')]:
            path=ROOT/name
            parts=[s for s in path.read_text().split('\n\n') if not s.startswith(('**AWS addressed-capacity acquisition','Current AWS addressed-capacity acquisition'))]
            if name.startswith('experiments/'):path.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:path.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md')
        command('git','commit','-m','Record completed addressed-capacity acquisition development')
        command('.venv-docker/bin/python','report/make_pdf.py')
        command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish completed capacity development evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
