"""Publish completed pool-clock development; pending scores stay absent."""
import fcntl
import hashlib
import json
from continue_unique_language import ROOT, command


def main():
    packet=ROOT/'experiments/results/diagnostics/aws_capacity_clock_decision_20261012_v1.json'
    d=json.loads(packet.read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    text='Cardinality-normalized pool8 clocks complete both acquisition sizes. '
    for level in d['levels']:
        r,raw,c=level['rows']
        text+=(f"At{c['targets']}targets, pool4/raw8/calibrated8 selected DEV{r['selected_nll']:.6f}/{raw['selected_nll']:.6f}/{c['selected_nll']:.6f}, "
            f"complete training{r['training_s']:.3f}/{raw['training_s']:.3f}/{c['training_s']:.3f}s. ")
    text+=d['scope']+' The contract preserves pool4 execution and calibrated cached/full-prefix gradients; common-clock finite-difference error9.91e-13. Retain the strongest completed quality/work member; no pending seed score or extrapolated exponent.'
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS capacity-clock calibration — 12 October.** '),('report/I_SCIENCE.md','Current AWS capacity-clock calibration. ')]:
            path=ROOT/name;parts=[s for s in path.read_text().split('\n\n') if not s.startswith(('**AWS capacity-clock calibration','Current AWS capacity-clock calibration'))]
            if name.startswith('experiments/'):path.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:path.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record completed capacity-clock quality and work comparison')
        command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish completed capacity-clock evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
