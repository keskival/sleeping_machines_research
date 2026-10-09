"""Replay completed B10 fits on their exact validation windows; TEST stays sealed."""
import argparse
import hashlib
import json
import resource
import socket
import time
from pathlib import Path

import numpy as np
import torch
import b10_tpp as protocol
from b10_components import components

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with (ROOT / path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    parser.add_argument('--fit-manifest', required=True); args = parser.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    output = ROOT / 'experiments/results/market' / f'{args.tag}.json'
    if output.exists():
        raise ValueError('Completed analysis exists; preserve it')
    manifest = json.loads((ROOT / args.fit_manifest).read_text())
    jobs = [j for j in manifest['jobs'] if j['driver'] == 'experiments/market/b10_tpp.py']
    if len(jobs) != 3:
        raise ValueError('Exactly the three admitted B10 fit records are required')
    records, inputs = [], {}
    for job in jobs:
        for path, expected in job['sources'].items():
            if digest(path) != expected:
                raise ValueError('Pinned fitted source changed: ' + path)
        record = json.loads((ROOT / job['result']).read_text())
        if record['status'] != 'completed' or record['tag'] != job['tag']:
            raise ValueError('Incomplete or mismatched fit')
        if record['source_sha256'] != job['sources'][job['driver']] or record['args']['score_test']:
            raise ValueError('Development source/protocol mismatch')
        inputs[job['result']] = digest(job['result']); records.append(record)
    if {r['model'] for r in records} != {'race', 'hawkes', 'poisson'}:
        raise ValueError('Registered model set differs')
    if len({r['train_windows'] for r in records}) != 1 or len({r['args']['train_every'] for r in records}) != 1:
        raise ValueError('Different training-window protocol')
    training = [w for day in protocol.TRAIN for w in protocol.windows(day, records[0]['args']['train_every'])]
    gaps = np.concatenate([np.diff(w[0]) for w in training]); positive = gaps[gaps > 0]; del training, gaps
    labels = ['zero', 'positive_lt_10us', '10us_to_1ms', '1ms_to_100ms', '100ms_or_longer']
    edges = [0, 10e-6, .001, .1]
    rows = {}; start = time.time()
    for record in records:
        model = {'poisson': lambda: protocol.Poisson([1.] * 6),
                 'hawkes': lambda: protocol.Hawkes([1.] * 6),
                 'race': lambda: protocol.Race(positive, record['args']['d'], record['args']['n_ln'])}[record['model']]()
        checkpoint = f"experiments/results/market/{record['tag']}.pt"
        inputs[checkpoint] = digest(checkpoint)
        model.load_state_dict(torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True)); model.eval()
        groups, days = {}, {}

        def accumulate(key, timing, mark, selected):
            count = int(selected.sum())
            if count:
                row = groups.setdefault(key, [0, 0., 0.]); row[0] += count
                row[1] += float(timing[selected].sum()); row[2] += float(mark[selected].sum())

        with torch.no_grad():
            for day in protocol.VAL:
                path = f'data/binance/npz/{day}.npz'; inputs[path] = digest(path)
                count, total_time, total_mark = 0, 0., 0.
                windows = protocol.windows(day, 10)
                for t, marks in protocol.batches(windows, 32):
                    timing, mark = components(model, t, marks)
                    timing, mark = timing[:, protocol.CTX - 1:], mark[:, protocol.CTX - 1:]
                    gap = (t[:, 1:] - t[:, :-1])[:, protocol.CTX - 1:]
                    current = marks[:, protocol.CTX:]; previous = marks[:, protocol.CTX - 1:-1]
                    count += timing.numel(); total_time += float(timing.sum()); total_mark += float(mark.sum())
                    accumulate('all', timing, mark, torch.ones_like(gap, dtype=torch.bool))
                    for index, label in enumerate(labels):
                        selected = gap == 0 if index == 0 else ((gap > 0) & (gap < edges[1]) if index == 1 else
                                   ((gap >= edges[index - 1]) & (gap < edges[index]) if index < 4 else gap >= edges[3]))
                        accumulate('gap:' + label, timing, mark, selected)
                    for label in range(6):
                        accumulate('mark:' + str(label), timing, mark, current == label)
                    accumulate('same_side', timing, mark, current // 3 == previous // 3)
                    accumulate('changed_side', timing, mark, current // 3 != previous // 3)
                days[str(day)] = dict(targets=count, time_ll=total_time / count, mark_ll=total_mark / count,
                                     total_ll=(total_time + total_mark) / count)
        formatted = {key: dict(targets=n, time_ll=t / n, mark_ll=m / n, total_ll=(t + m) / n)
                     for key, (n, t, m) in groups.items()}
        reproduction = abs(formatted['all']['total_ll'] - record['best_val_ll_per_event'])
        if reproduction > 1e-7:
            raise ValueError('Saved validation likelihood does not reproduce: ' + record['tag'])
        rows[record['model']] = dict(tag=record['tag'], groups=formatted, days=days,
                                     saved_validation_reproduction_error=reproduction)
    contrasts = {key: dict(targets=rows['race']['groups'][key]['targets'],
                          time_delta=rows['race']['groups'][key]['time_ll'] - rows['hawkes']['groups'][key]['time_ll'],
                          mark_delta=rows['race']['groups'][key]['mark_ll'] - rows['hawkes']['groups'][key]['mark_ll'],
                          total_delta=rows['race']['groups'][key]['total_ll'] - rows['hawkes']['groups'][key]['total_ll'])
                 for key in rows['race']['groups']}
    sources = [Path(__file__).resolve(), Path(__file__).with_name('b10_components.py'),
               Path(__file__).with_name('b10_tpp.py'), ROOT / 'experiments/tpp/race_tpp_v5.py']
    result = dict(status='completed', tag=args.tag, battle='B10', evidence_level='first-pass development diagnostics',
                  rows=rows, race_minus_hawkes=contrasts, inputs=inputs,
                  scoring='exact registered validation windows; 896 scored targets/window; nats/scored target',
                  test='sealed; no test data accessed', wall_s=time.time() - start,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hardware=dict(host=socket.gethostname(), threads=torch.get_num_threads()),
                  source_sha256={str(p.relative_to(ROOT)): digest(str(p.relative_to(ROOT))) for p in sources})
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(contrasts['all']), flush=True)


if __name__ == '__main__':
    main()
