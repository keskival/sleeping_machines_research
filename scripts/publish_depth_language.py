"""Publish only the completed registered depth/exposure comparison."""
import fcntl
import json
import time
from continue_unique_language import ROOT, command


def main():
    packet=ROOT/'experiments/results/diagnostics/aws_integrated_depth_exposure_20261012_v1.json'
    deadline=time.monotonic()+3*3600
    while not packet.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(packet))
        time.sleep(10)
    d=json.loads(packet.read_text());assert d['status']=='completed'
    text='First-pass P24D4H2U4 depth/exposure development completed. '
    for level in d['levels']:
        r,c=level['rows']
        text+=(f"At {c['targets']} targets, selected DEV depth2/depth4 {r['selected_nll']:.6f}/{c['selected_nll']:.6f}; "
               f"final {r['final_nll']:.6f}/{c['final_nll']:.6f}; complete training {r['training_s']:.3f}/{c['training_s']:.3f}s. ")
    text+='Same proper tokens/prior/update counts; depth4 doubles selected writes and core receivers. One seed, full factual BP and temporal/counterfactual mechanisms retained. Source-bound aws_integrated_depth_exposure_20261012_v1. This member comparison is not a family verdict or learned-credit efficiency claim. Public validation and reserved4M untouched. Next engineering decision follows the completed quality/work and depth error analysis.'
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS integrated language depth/exposure pilot — 12 October.** '),
                            ('report/I_SCIENCE.md','Current AWS integrated language depth/exposure development. ')]:
            path=ROOT/name;parts=[s for s in path.read_text().split('\n\n') if not s.startswith(('**AWS integrated language depth/exposure','Current AWS integrated language depth/exposure'))]
            if name.startswith('experiments/'):path.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:path.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md')
        command('git','commit','-m','Record completed integrated depth/exposure development')
        command('.venv-docker/bin/python','report/make_pdf.py')
        command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish completed depth/exposure evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
