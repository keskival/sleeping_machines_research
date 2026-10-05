"""One guarded AWS token stage; adapts results to the existing slot publisher."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import runpy
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--stage', choices=['data', 'resume', 'fit'], required=True)
    p.add_argument('--require-gain-against', nargs=2, metavar=('CANDIDATE', 'CONTROL'))
    a, extra = p.parse_known_args()
    packet = json.loads(Path(a.manifest).read_text())
    pins = packet['token_source_sha256']
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source binding mismatch: ' + name)
    out = ROOT / 'experiments/results/token_language' / (a.tag + '.aws.json')
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.stage == 'data':
        for item in packet['data']:
            path = ROOT / item['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix('.download.tmp')
                with urllib.request.urlopen(item['url'], timeout=60) as response, open(temporary, 'wb') as f:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        f.write(block)
                if sha(temporary) != item['sha256']:
                    raise ValueError('Downloaded data hash mismatch: ' + str(path))
                temporary.replace(path)
            if sha(path) != item['sha256']:
                raise ValueError('Existing data hash mismatch: ' + str(path))
        result = dict(status='completed', stage='data', data=packet['data'])
    elif a.stage == 'resume':
        runpy.run_path(str(ROOT / 'experiments/aws_integrated_token_resume_contracts.py'), run_name='__main__')
        result = json.loads((ROOT / 'experiments/results/diagnostics/aws_integrated_token_resume_contract_20261005T181000Z_v1.json').read_text())
    else:
        if a.require_gain_against:
            candidate, control = [json.loads((ROOT / name).read_text()) for name in a.require_gain_against]
            if not candidate['best_dev_nll'] < control['best_dev_nll']:
                raise ValueError('Larger future-credit fit deferred: small candidate did not beat control')
        import integrated_token_language_lab as lab
        sys.argv = ['integrated_token_language_lab.py', '--tag', a.tag, *extra]
        lab.main()
        result = json.loads((out.parent / (a.tag + '.json')).read_text())
        result['final_weights'] = str((out.parent / (a.tag + '.pt')).relative_to(ROOT))
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source changed during execution: ' + name)
    result.update(source_sha256=pins, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
