"""Publish signed frozen-history evidence, without converting it into a benchmark claim."""
import fcntl,json,statistics,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
    path=ROOT/'experiments/results/diagnostics/aws_acquired_short_history_transfer_20261013_v1.json'
    queue.DEADLINE=time.monotonic()+3*3600
    d=queue.wait_result(path);assert d['status']=='completed' and len(d['paired'])==3
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert sha(ROOT/name)==digest,name
    lines=[]
    for span in ('1','4'):
        before=[r['history_value_before'][span] for r in d['paired']]
        after=[r['history_value_after'][span] for r in d['paired']]
        delta=[y-x for x,y in zip(before,after)]
        lines.append(f"History reset every{span}tokens: mean signed reset NLL penalty512k{statistics.mean(before):+.6f},8M{statistics.mean(after):+.6f}; acquisition/history interaction{statistics.mean(delta):+.6f}±{statistics.stdev(delta):.6f} (sampleSD).")
    text=' '.join(lines)+' '+d['scope']
    with open('/tmp/aws-language-publication.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS acquired short history — 13 October.** '),('report/I_SCIENCE.md','Current AWS acquired short history. ')]:
            p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith(prefix)]
            p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record signed acquired-history transfer diagnostic')
        command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish acquired-history evidence and current PDFs')
        command('git','pull','--rebase');command('git','push','origin','main')

if __name__=='__main__':main()
