"""Source-bound independent-seed confirmation of fixed-prior 8M acquisition."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--through-seed',type=int,choices=(7,8),required=True);a=p.parse_args()
    inputs,sources,rows={},{},[]
    def load(tag):
        path=ROOT/f'experiments/results/token_language/{tag}.json';blob=path.read_bytes();d=json.loads(blob)
        assert d['status']=='completed'
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
        for name,digest in d['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            assert name not in sources or sources[name]==digest
            sources[name]=digest
        return d
    recipe=None
    for seed in range(6,a.through_seed+1):
        before=load(f'aws_integrated_window_replay_512k_20261012_s{seed}_v1')
        current=load(f'aws_integrated_fresh_scaling_8m_pool4_{"20261012" if seed==6 else "20261013"}_s{seed}_v1')
        ignored=('tag','steps')
        assert {k:v for k,v in before['args'].items() if k not in ignored}=={k:v for k,v in current['args'].items() if k not in ignored}
        settings={k:v for k,v in current['args'].items() if k not in ('tag','seed')}
        if recipe is None:recipe=settings
        else:assert settings==recipe
        for name in ('train_sha256','dev_sha256'):assert before['identity'][name]==current['identity'][name]
        def canon(r):return {k:v for k,v in r.items() if k not in ('wall_s','learner_fingerprint')}
        assert [canon(r) for r in current['curve'] if r['step']<=1024]==[canon(r) for r in before['curve']]
        assert current['presentations_total']==8388352
        at2m=next(r['dev_nll'] for r in current['curve'] if r['step']==4096)
        final=current['curve'][-1]['dev_nll'];ms=1000*current['train_wall_s_total']/current['presentations_total']
        rows.append(dict(seed=seed,targets=current['presentations_total'],final_nll=final,selected_nll=current['best_dev_nll'],
            nll_at_512k=before['curve'][-1]['dev_nll'],nll_at_2m=at2m,
            gain_512k_to_8m=before['curve'][-1]['dev_nll']-final,gain_2m_to_8m=at2m-final,
            training_s=current['train_wall_s_total'],ms_per_target=ms,max_rss_kb=current['max_rss_kb'],
            exact_512k_acquisition_prefix=True))
    summary={}
    for field in ('final_nll','selected_nll','gain_512k_to_8m','gain_2m_to_8m','training_s','ms_per_target'):
        vals=[r[field] for r in rows];summary[field+'_mean']=statistics.mean(vals);summary[field+'_sd']=statistics.stdev(vals)
    source=str(Path(__file__).relative_to(ROOT));sources[source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result=dict(status='completed',battle='R1',rows=rows,summary=summary,
        continue_confirmation=all(r['gain_2m_to_8m']>.05 and r['ms_per_target']<1.5 and r['max_rss_kb']<1500000 for r in rows),
        input_sha256=inputs,source_sha256=sources,
        scope='Independent seeds, same fixed262144-token TRAIN prior corpus, GPT2 FineWeb fresh data, integrated P24D2H2U4/credit16 and DEV2048 interval. Each earlier acquired512k trajectory matches exactly. Final DEV at512k/2M/8M measures acquisition; selected DEV has increasing checkpoint-selection opportunities. Complete whole-fit CPU training measured; sample SD. Public/reserved data untouched; no external comparison, exponent or nonstationarity claim.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
