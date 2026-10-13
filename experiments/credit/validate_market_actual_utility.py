"""Read-only verification of completed B10 actual-utility artifacts."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    packet=json.loads((ROOT/'experiments/results/market/curie_b10_actual_utility_20261013_v1_validation.json').read_text())
    for name,digest in packet['input_sha256'].items():
        path=ROOT/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        result=json.loads(path.read_text());assert result['status']=='completed'
        for key in ('source_sha256','input_sha256'):
            for source,expected in result.get(key,{}).items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
        if 'dev' in result:
            dev=result['dev'];assert len(dev['rows'])*16==dev['targets'];assert math.isfinite(dev['nll'])
            assert dev['teacher_future_loss_max_error']<1e-7
    print('Completed source-bound actual-utility artifacts verified; online final-print failure recorded.')
if __name__=='__main__':main()
