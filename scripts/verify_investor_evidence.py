"""Verify investor/report evidence against committed result files.

1. Every evidence packet investment/*.json: each 64-hex binding under a *sha256* key
   must match the committed file (missing files and mismatches are reported).
2. investment/claims_registry.json: each headline value is recomputed from completed
   result files (status completed where present), must round to the displayed text at
   the stated precision, and the displayed text must occur in every listed document.
Exit status 1 on any failure. Writes a JSON report with --out.
"""
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def packets():
    rows = []
    for f in sorted(glob.glob(str(ROOT / 'investment/*.json'))):
        name = os.path.relpath(f, ROOT)
        if name.endswith('claims_registry.json'): continue
        data = json.load(open(f))
        for key, bindings in data.items():
            if 'sha256' not in key or not isinstance(bindings, dict): continue
            for path, digest in bindings.items():
                if not (isinstance(digest, str) and len(digest) == 64): continue
                if not (ROOT / path).exists(): rows.append(dict(packet=name, file=path, status='missing'))
                else: rows.append(dict(packet=name, file=path, status='ok' if sha(path) == digest else 'mismatch'))
    return rows


def resolve(spec):
    if 'constant' in spec: return float(spec['constant'])
    if 'mean' in spec:
        values = [resolve(s) for s in spec['mean']]; return sum(values) / len(values)
    if 'ratio' in spec:
        a, b = (resolve(s) for s in spec['ratio']); return a / b
    data = json.loads((ROOT / spec['file']).read_text())
    if data.get('status', 'completed') != 'completed': raise ValueError(f"{spec['file']} is not completed")
    for step in spec['path']: data = data[step]
    return float(data)


def claims():
    registry = json.loads((ROOT / 'investment/claims_registry.json').read_text()); rows = []
    for c in registry['claims']:
        row = dict(id=c['id'], display=c['display'])
        try:
            value = resolve(c['value']); row['value'] = value
            row['value_matches'] = f"{value:.{c['decimals']}f}" == c['display']
        except Exception as error:
            row['error'] = repr(error); row['value_matches'] = False
        row['missing_from'] = [d for d in c['documents'] if c['display'] not in (ROOT / d).read_text()]
        row['status'] = 'ok' if row['value_matches'] and not row['missing_from'] else 'fail'
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out'); a = p.parse_args()
    pk, cl = packets(), claims()
    summary = dict(packet_bindings=len(pk), packet_ok=sum(r['status'] == 'ok' for r in pk),
                   packet_problems=[r for r in pk if r['status'] != 'ok'],
                   claims=len(cl), claims_ok=sum(r['status'] == 'ok' for r in cl), claim_failures=[r for r in cl if r['status'] != 'ok'])
    headline_path=ROOT/'investment/public_benchmark_headlines.json'
    headlines=[]
    if headline_path.exists():
        from public_headline_consistency import check
        headlines=check(ROOT,json.loads(headline_path.read_text()))
        summary['headline_documents']=len(headlines)
        summary['headline_failures']=[r for r in headlines if r['status']!='ok']
    # Readable prose (founder, 16 Oct: "weird garbage"): investor-facing text must not run words together. The repair is
    # scripts/fix_prose_spacing.py; publication scripts must write spaced, plain sentences.
    import importlib.util, re
    spec = importlib.util.spec_from_file_location('fps', ROOT/'scripts/fix_prose_spacing.py'); prose = []
    rx = re.compile(r'[a-z\)][;,][A-Za-z]|\b[a-z]{3,}\d|\d[a-z]{4,}\b')
    for doc in ('investment/PITCH.md', 'investment/INVESTMENT_CASE.md', 'investment/INVESTOR_PROOF_PLAN.md', 'investment/brief/evidence_brief.tex'):
        n = 0
        for line in (ROOT/doc).read_text().splitlines():
            st = line.lstrip()
            if st.startswith(('%', '|---')) or (st.startswith('\\') and not st.startswith(('\\textbf', '\\emph'))) or '\\foreach' in st or '\\newcommand' in st: continue
            line = re.sub(r'`[^`]*`|\\(path|texttt|url)\{[^}]*\}|https?://\S+|\]\([^)]*\)|\$[^$]*\$|\b(text8|enwik8|MIMIC2|GPT2|S2P2)\b', ' ', line)
            n += len(rx.findall(line))
        if n: prose.append(dict(document=doc, run_together_spots=n))
    summary['prose_failures'] = prose
    print(json.dumps(summary, indent=1))
    if a.out: Path(a.out).write_text(json.dumps(dict(summary=summary, packets=pk, claims=cl, headlines=headlines), indent=1) + '\n')
    sys.exit(0 if not summary['claim_failures'] and not summary.get('headline_failures') and not summary.get('prose_failures') and not [r for r in pk if r['status'] == 'mismatch'] else 1)


if __name__ == '__main__':
    main()
