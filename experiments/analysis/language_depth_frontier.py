"""R1 saved-result comparison; no fitting or additional evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=('targets','target_start','target_end_inclusive','window','lanes','eos_state_reset')
BASIC='aws_bernoulli_teacher_8m_20261013_s6_v2'
STRONG='aws_multidepth_readout_8m_20261013_s6_v1'

def build(root=ROOT):
    sources={}
    def read(path):
        p=root/path
        sources[path]=hashlib.sha256(p.read_bytes()).hexdigest()
        r=json.loads(p.read_text())
        if r.get('status')!='completed':raise ValueError(f'Incomplete evidence: {path}')
        return r
    refs=read('experiments/results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json')
    scores={BASIC:refs['rows'][0]['scored'],STRONG:refs['rows'][1]['scored']}
    deep=read('experiments/results/diagnostics/aws_native_deep_d8_8m_quality_20261013_s6_v1.json')
    for r in deep['rows']:scores[r['variant']]=r['scored']
    for p in sorted((root/'experiments/results/diagnostics').glob('curie*_quality.json')):
        r=json.loads(p.read_text())
        if r.get('status')=='completed' and 'long_history_dev' in r:
            read(str(p.relative_to(root)))
            scores[p.stem.removesuffix('_quality')]=r['long_history_dev']
    protocol={k:scores[STRONG][k] for k in PROTOCOL}
    rows=[];excluded=[]
    for tag,score in scores.items():
        path=f'experiments/results/token_language/{tag}.json'
        if not (root/path).exists():continue
        if {k:score.get(k) for k in PROTOCOL}!=protocol:
            excluded.append(dict(tag=tag,reason='Different scored targets/history/reset protocol'));continue
        fit=read(path)
        rows.append(dict(tag=tag,nll=score['nll'],parameters=fit['parameters'],
            acquired_targets=fit['presentations_total'],training_s=fit['train_wall_s_total'],
            training_s_per_target=fit['train_wall_s_total']/fit['presentations_total'],
            nll_gain_over_strong_d2=scores[STRONG]['nll']-score['nll'],
            process_wall_precision='lower bound' if 'whole_fit_process_wall_lower_bound_s' in fit else 'per-invocation only'))
    for row in rows:
        row['quality_training_time_frontier']=not any(other['nll']<=row['nll'] and other['training_s']<=row['training_s'] and
            (other['nll']<row['nll'] or other['training_s']<row['training_s']) for other in rows)
    return dict(status='completed',battle='R1',protocol=protocol,rows=rows,excluded=excluded,source_sha256=sources,
        scope='Saved single-seed development evidence; matching targets/history, varying architectures/acquisition/teaching. Training seconds are measured successful-trajectory time, not FLOPs or whole process cost. Frontier is descriptive across hosts, not a matched-compute win. Public validation untouched.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    with Path(a.out).open('x') as f:json.dump(build(),f,indent=2);f.write('\n')
