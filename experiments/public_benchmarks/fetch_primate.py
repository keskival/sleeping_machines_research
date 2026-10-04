"""Stream official data bytes and verify vendored MD5s; no numerical parsing."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOADER = ROOT/'experiments/vendor/neurobench_2_3_0/neurobench/datasets/primate_reaching.py'
SESSIONS = ('indy_20160622_01', 'indy_20160630_01', 'indy_20170131_02',
            'loco_20170210_03', 'loco_20170215_02', 'loco_20170301_05')


def main():
    tree = ast.parse(LOADER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assignments = {t.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name) and t.id in ('url', 'md5s')}
    output = ROOT/'data/neurobench/primate_reaching'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for session in SESSIONS:
        name = session+'.mat'
        target = output/name
        expected = assignments['md5s'][name]
        url = assignments['url']+name
        temporary = target.with_suffix('.download_20261004T150000Z.part')
        if not target.exists():
            assert not temporary.exists(), 'Preserve incomplete download; new recovery required'
            request = urllib.request.Request(url, headers={'User-Agent': 'SleepingMachines-data-verification/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            source = temporary
        else:
            source = target
        md5 = hashlib.md5()
        sha = hashlib.sha256()
        with source.open('rb') as data:
            while chunk := data.read(1024*1024):
                md5.update(chunk)
                sha.update(chunk)
        assert md5.hexdigest() == expected, name
        if source == temporary:
            temporary.rename(target)
        record = dict(session=session, path=str(target.relative_to(ROOT)), url=url,
                      bytes=target.stat().st_size, md5=expected, sha256=sha.hexdigest())
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest = ROOT/'experiments/results/diagnostics/aws_primate_data_20261004T150000Z.json'
    assert not manifest.exists()
    manifest.write_text(json.dumps(dict(status='verified_data_bytes', files=records,
                        loader_sha256=hashlib.sha256(LOADER.read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        scope='Official bytes and checksums only; no arrays loaded, training or scoring.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
