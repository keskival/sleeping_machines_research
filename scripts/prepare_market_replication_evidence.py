"""Bind completed three-seed trade-event acquisition evidence; no fitting or test scoring."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    bindings = {}
    for family in ('native', 'lstm', 'transformer'):
        values = []
        for seed in range(3):
            suffix = '20261014_v1' if seed == 0 or (family == 'native' and seed == 1) else ('20261015_v2' if family == 'lstm' and seed == 2 else '20261015_v1')
            path = Path(f'experiments/results/market/aws_b10_stable_{family}_s{seed}_{suffix}.json')
            raw = (ROOT / path).read_bytes(); d = json.loads(raw)
            assert d['status'] == 'completed' and d['family'] == family and d['seed'] == seed
            assert not d['september_opened']
            values.append(d['final_dev']['nll'])
            bindings[str(path)] = hashlib.sha256(raw).hexdigest()
        rows.append(dict(family=family, seeds=[0, 1, 2], dev_nll=values, mean_dev_nll=sum(values)/3))
    native = rows[0]['dev_nll']
    assert all(native[i] < row['dev_nll'][i] for row in rows[1:] for i in range(3))
    packet = dict(status='completed', evidence_level='three-seed development acquisition result', rows=rows,
                  scope='Fixed-weight August development evaluation of the acquired tape models under the source-bound stable-head budget admission. Native leads each reference at each paired seed. Different from online October confirmation and price forecasting; no additional test or online confirmation inferred. All inherited, interruption and selection work remains in original receipts.',
                  next_decision='Reuse the completed learners for frozen-protocol online August/October replication; no new offline reference fit.', input_sha256=bindings)
    (ROOT/'investment/market_replication_evidence.json').write_text(json.dumps(packet, indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
