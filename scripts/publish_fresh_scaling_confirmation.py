"""Publish completed independent-seed acquisition evidence."""
import argparse
import fcntl
import hashlib
import json
from continue_unique_language import ROOT,command


def main():
    p=argparse.ArgumentParser();p.add_argument('--through-seed',type=int,choices=(7,8),required=True);a=p.parse_args()
    d=json.loads((ROOT/f'experiments/results/diagnostics/aws_fresh_scaling_8m_confirmation_20261013_s{a.through_seed}_v1.json').read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    s=d['summary'];text=(f"Completed{len(d['rows'])}independent seeds at8388352targets. Final DEV{s['final_nll_mean']:.6f}±{s['final_nll_sd']:.6f}; "
        f"paired final-NLL gain over2M{s['gain_2m_to_8m_mean']:.6f}±{s['gain_2m_to_8m_sd']:.6f}, "
        f"over512k{s['gain_512k_to_8m_mean']:.6f}±{s['gain_512k_to_8m_sd']:.6f}. "
        f"Complete training{s['training_s_mean']:.3f}±{s['training_s_sd']:.3f}s; "
        f"{s['ms_per_target_mean']:.4f}ms/target. "+d['scope'])
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS acquisition confirmation — 13 October.** '),('report/I_SCIENCE.md','Current AWS acquisition confirmation. ')]:
            path=ROOT/name;parts=[x for x in path.read_text().split('\n\n') if not x.startswith(('**AWS acquisition confirmation','Current AWS acquisition confirmation'))]
            if name.startswith('experiments/'):path.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:path.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record independent-seed large acquisition confirmation')
        command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish integrated acquisition confirmation')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
