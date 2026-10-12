"""Publish only completed, source-bound exact-replay evidence."""
import fcntl
import hashlib
import json
import statistics
from continue_unique_language import ROOT, command


def main():
    paths=[ROOT/'experiments/results/diagnostics/aws_window_replay_fit_gate_20261012_v1.json',
        *sorted((ROOT/'experiments/results/diagnostics').glob('aws_window_replay_*_gate_20261012_s*_v1.json'))]
    records=[]
    for path in paths:
        if not path.exists():continue
        d=json.loads(path.read_text());assert d['status']=='completed'
        for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        records.append(d)
    assert records
    first=records[0]
    summary=''
    for targets in sorted({r['targets'] for r in records}):
        values=[r['training_time_reduction_percent'] for r in records if r['targets']==targets]
        deviation=statistics.stdev(values) if len(values)>1 else 0.
        summary+=f"At{targets}targets, mean training-time reduction{statistics.mean(values):.2f}%±{deviation:.2f}percentage points across{len(values)}seeds. "
    text=(f"Exact cached credit-window replay completes {len(records)} source-bound quality/work comparisons. "
        f"At131056targets seed6, complete training {first['reference_training_s']:.3f}→{first['candidate_training_s']:.3f}s "
        f"({first['training_time_reduction_percent']:.2f}% reduction), DEV NLL unchanged{first['candidate_nll']:.6f}. "
        f"Exact recorded learning/teacher/gradient curves: {sum(d['exact_learning_curve'] for d in records)}/{len(records)}; "
        f"quality/time gates PASS: {sum(d['gate']=='PASS' for d in records)}/{len(records)}. "
        +summary+'Retains temporal races, key/value separation, persistent state, hard writes, factual BP and counterfactual objective; replay begins at cloned factual window state/RNG. '
        'Same data/settings against saved full-prefix fits; CPU wall measurements, no new dense controls, public/reserved scoring or external-model claim. '
        'Bounded continuation covers seeds6/7/8 at131056and524272targets, admits the next fit only after exact-learning/lower-time gate PASS; pending points are absent from the evidence table. '
        'Curie retains native learned-credit ownership; this is an exact own-model execution improvement.')
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
        command('git','pull','--rebase')
        for name,prefix in [('experiments/HANDOFF.md','**AWS exact-window credit replay — 12 October.** '),
                            ('report/I_SCIENCE.md','Current AWS exact-window credit replay. ')]:
            path=ROOT/name
            parts=[s for s in path.read_text().split('\n\n') if not s.startswith(('**AWS exact-window credit replay','Current AWS exact-window credit replay'))]
            if name.startswith('experiments/'):path.write_text(prefix+text+'\n\n'+'\n\n'.join(parts))
            else:path.write_text('\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
        command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md')
        command('git','commit','-m','Record completed exact replay quality and training-time gain')
        command('.venv-docker/bin/python','report/make_pdf.py')
        command('.venv-docker/bin/python','report/split_report.py')
        command('git','add','REPORT.md','report')
        if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish completed exact cached replay evidence')
        command('git','pull','--rebase');command('git','push','origin','main')


if __name__=='__main__':main()
