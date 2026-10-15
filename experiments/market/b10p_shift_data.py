"""Acquire one fresh October day for the B10-P price confirmation, only after its preregistration is committed.
Downloads the checksum-verified Binance archive and writes t/mark/price/quantity arrays; never overwrites; September
is never opened."""
import argparse, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/market'))
from prepare import convert, RAW, OUT

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--preregistration', required=True); a = p.parse_args()
    prereg = ROOT/a.preregistration; spec = json.loads(prereg.read_text()); day = spec['day']; assert day.startswith('2026-10-'), day
    # The preregistration must be committed and unmodified before any byte of the new day is fetched.
    assert subprocess.run(['git', 'ls-files', '--error-unmatch', a.preregistration], cwd=ROOT, capture_output=True).returncode == 0
    assert subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', a.preregistration], cwd=ROOT).returncode == 0
    out = ROOT/f'experiments/results/market/{a.tag}.json'; assert not out.exists(); assert not (OUT/f'{day}.npz').exists() or 'p' in __import__('numpy').load(OUT/f'{day}.npz').files
    name = f'BTCUSDT-aggTrades-{day}.zip'
    for suffix in ('.CHECKSUM', ''):
        path = RAW/(name + suffix)
        if not path.exists():
            tmp = path.with_name(path.name + '.downloading'); urllib.request.urlretrieve('https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/' + name + suffix, tmp); tmp.replace(path)
    summary = convert(day) if not (OUT/f'{day}.npz').exists() else None
    record = dict(status='completed', battle='B10', task='B10-P price confirmation data', day=day, summary=summary, archive_sha256=sha(RAW/name),
                  npz_sha256=sha(OUT/f'{day}.npz'), preregistration_sha256=sha(prereg), source_sha256={n: sha(ROOT/n) for n in ['experiments/market/b10p_shift_data.py', 'experiments/market/prepare.py']},
                  september_opened=False)
    out.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
