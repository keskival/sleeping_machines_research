"""Publish completed frozen final-weight development transfer."""
import fcntl
import json
from continue_unique_language import ROOT,command
from continue_window_replay import sha


def main():
    path=ROOT/'experiments/results/diagnostics/aws_fresh_scaling_transfer_20261013_v1.json'
    d=json.loads(path.read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert sha(ROOT/name)==digest,name
    text=(f"Frozen final weights scored on32640new development targets across three seeds. Paired512k-to8M acquired-data NLL gain "
          f"{d['paired_gain_mean']:+.6f}±{d['paired_gain_sd']:.6f} (sampleSD across seeds). "+d['scope'])
    with open('/tmp/aws-language-publication.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS frozen acquisition transfer — 13 October.** '),('report/I_SCIENCE.md','Current AWS frozen acquisition transfer. ')]:
            p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith(('**AWS frozen acquisition transfer','Current AWS frozen acquisition transfer'))]
            if name.startswith('experiments/'):p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:p.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record frozen acquired-learning transfer on independent development interval')
        command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish frozen acquisition transfer evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
