"""Read completed fixed-protocol batches; never score data or select a mode."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'aws_mg_tau17_r1_p16d2_k8_delta_mix8_from'
SUFFIX = '_20261004T061500Z.json'


def collect():
    parents = []
    rows = []
    settings = None
    sources = None
    data_sha = None
    for start in (0, 10, 20):
        path = ROOT/'experiments/results/neurobench_mg'/f'{PREFIX}{start}{SUFFIX}'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        assert r['status'] == 'completed' and r['primary_mode'] == 'mix8'
        assert r['args']['tau'] == 17 and r['args']['repeats'] == 10
        assert r['args']['first_repeat'] == start
        assert [v['repeat'] for v in r['repeats']] == list(range(start, start+10))
        current = {k: v for k, v in r['args'].items() if k not in ('tag', 'first_repeat')}
        if settings is None:
            settings, sources, data_sha = current, r['source_sha256'], r['data_sha256']
        assert current == settings and r['source_sha256'] == sources and r['data_sha256'] == data_sha
        for row in r['repeats']:
            assert row['smape'] == row['smape_by_mode']['mix8']
        rows.extend(r['repeats'])
        parents.append(dict(path=str(path.relative_to(ROOT)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len({r['repeat'] for r in rows}) == len(rows)
    complete = len(rows) == 30
    if complete:
        assert sorted(r['repeat'] for r in rows) == list(range(30))
    modes = ('mix8', 'argmax', 'sampled')
    return dict(status='completed_protocol' if complete else 'partial_protocol',
                completed_repeats=len(rows), required_repeats=30, primary_mode='mix8',
                mean_smape_by_mode={m: statistics.mean(r['smape_by_mode'][m] for r in rows)
                                    for m in modes} if rows else {},
                primary_repeat_stddev=statistics.stdev(r['smape'] for r in rows) if len(rows)>1 else None,
                protocol_claim_eligible=complete, parents=parents, settings=settings,
                source_sha256=sources, data_sha256=data_sha,
                scope='Fixed primary mix8. Alternative modes descriptive only. Partial batches cannot be compared to the full leaderboard. Repeats use overlapping windows of one series; their dispersion is not an independent-sample confidence interval. No measured serving/resource win.',
                collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    result = collect()
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'completed_repeats', 'mean_smape_by_mode')}))
