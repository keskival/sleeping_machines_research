"""Insert the 7 October public-benchmark headline page after the cover of report/sleeping_machines_status.pdf; every
previous page is retained identically (same convention as publish_unification_headlines_20261006_v1.py)."""
import hashlib, html, json, resource, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
resource.setrlimit(resource.RLIMIT_AS, (1000000*1024, 1000000*1024))
started = time.monotonic()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
def peak():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))
assert available() >= 8192*1024
packet = ROOT/'report/public_wins_headline_evidence_20261007_v1.json'
e = json.loads(packet.read_text())
assert e['status'] == 'completed'
for source, digest in e['input_sha256'].items(): assert sha(ROOT/source) == digest, source
folder = ROOT/'report/archive/public_wins_20261007_v1'
folder.mkdir(parents=True, exist_ok=False)
pdf = ROOT/'report/sleeping_machines_status.pdf'
previous = sha(pdf)
shutil.copy2(pdf, folder/'previous.pdf')
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
import pymupdf
body = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=12)
cell = ParagraphStyle('cell', fontName='Helvetica', fontSize=8, leading=10.5)
heading = ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=18, leading=23)
def p(text, style=body): return Paragraph(html.escape(text), style)
t, c = e['easytpp'], e['clinical']
tx, tb, so, am = t['taxi'], t['taobao'], t['stackoverflow'], t['amazon']
p19, p12 = c['p19'], c['p12']
rows = [
    ['Benchmark (public, official splits)', 'Ours (sealed test)', 'Best published', 'Verdict and compute'],
    ['EasyTPP Taxi (nats/event, higher better)', f"{tx['ours']['mean']:.4f} ± {tx['ours']['sd']:.4f}; 5-seed mixture {tx['ours']['mixture']:.4f}",
     f"{tx['published']['ll']:.3f} ± {tx['published']['sd']:.3f} (S2P2)", f"Win, 5 seeds. Single model at 1/12 of S2P2's parameters and compute; mixture beats every published model at 0.41×."],
    ['EasyTPP Taobao', f"{tb['ours']['mean']:.4f} ± {tb['ours']['sd']:.4f}", f"{tb['published']['ll']:.3f} ± {tb['published']['sd']:.3f} (IFTPP)",
     'Win, 5 seeds: +0.081 nats/event at 0.92× S2P2 per-event compute.'],
    ['EasyTPP StackOverflow', f"{so['ours_matched']['mean']:.4f} ± {so['ours_matched']['sd']:.4f} (matched size)",
     f"{so['published']['ll']:.3f} ± {so['published']['sd']:.3f} (S2P2)", 'Win at matched compute, 5 seeds: 0.968× parameters, 1.015× per-event compute.'],
    ['EasyTPP Amazon', f"{am['ours']['mean']:.4f} ± {am['ours']['sd']:.4f}", f"{am['published']['ll']:.3f} (S2P2)",
     'Mean ahead at 0.29× compute; not confirmed (one low-basin seed).'],
    ['P19 sepsis (ICU records)', f"AUPRC {p19['auprc'][0]:.3f} ± {p19['auprc'][1]:.3f}; AUROC {p19['auroc'][0]:.3f} ± {p19['auroc'][1]:.3f}",
     f"AUPRC {p19['published']['auprc'][0]:.3f}; AUROC {p19['published']['auroc'][0]:.3f} (MTM)", 'Win, five splits: AUPRC beyond both split spreads; 62,681 parameters.'],
    ['P12 mortality (ICU records)', f"AUPRC {p12['auprc'][0]:.3f}; AUROC {p12['auroc'][0]:.3f} ± {p12['auroc'][1]:.3f}",
     f"AUPRC {p12['published']['auprc']:.3f}; AUROC {p12['published']['auroc']:.3f} (MTM)", 'Behind MTM on AUROC, level on AUPRC; ahead of all other published models.'],
]
flow = [p('Public benchmarks: wins in two event-data domains (7 October 2026)', heading), Spacer(1,10),
    p('One model family — persistent memories that decay and rotate with elapsed time, sparse addressed writes and temporal races — now wins public leaderboards of two kinds: temporal point processes (EasyTPP, ICLR 2024) and irregular clinical records (P19). Every row is a sealed test on the official splits under a pre-registered protocol.'), Spacer(1,10)]
tab = Table([[p(v, cell) for v in r] for r in rows], colWidths=[100,125,100,165], repeatRows=1)
tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3fb')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
flow += [tab, Spacer(1,12),
    p('Three of five EasyTPP datasets are won, two at a fraction of the state of the art\'s compute and one at matched compute; the P19 classifier uses the same temporal memory layer. The Taxi, Taobao and StackOverflow wins pass the recording-grid audit and hold under the EasyTPP/S2P2 Monte Carlo estimator.'), Spacer(1,8),
    p(f"In development, reported as such: Retweet (TEST protocol), PAM, and the sealed FAS v2 benchmark on anonymous interleaved logs ({e['fas_v2']})."), Spacer(1,8),
    p('Source-bound packet: report/public_wins_headline_evidence_20261007_v1.json (own numbers computed from the result files; published references and compute ratios from experiments/B1_EASYTPP.md and experiments/B2_IRREGULAR_TS.md). All earlier pages, including the 6 October cross-domain page, are retained.')]
appendix = folder/'headline_page.pdf'
SimpleDocTemplate(str(appendix), pagesize=A4, leftMargin=45, rightMargin=45, topMargin=40, bottomMargin=40).build(flow)
temporary = folder/'published.pdf'
with pymupdf.open(pdf) as old, pymupdf.open(appendix) as add:
    old_count = len(old)
    assert len(add) == 1, len(add)
    old.insert_pdf(add, start_at=1)
    old.save(temporary, garbage=0, deflate=False)
with pymupdf.open(temporary) as new, pymupdf.open(folder/'previous.pdf') as old:
    assert len(new) == len(old)+1
    for i in range(len(old)): assert new[i if i == 0 else i+1].get_text() == old[i].get_text()
    text = new[1].get_text()
    assert all(v in text for v in ('0.5250', '1.3991', '-2.1525'.replace('-', '-'), '0.639', 'Win at matched compute')), text[:400]
    for x0,y0,x1,y1,*_ in new[1].get_text('blocks'):
        assert min(x0,y0) >= 0 and x1 <= new[1].rect.width and y1 <= new[1].rect.height
    new[1].get_pixmap().save(str(folder/'preview.png'))
assert peak() < 300000 and available() >= 8192*1024 and sha(pdf) == previous
shutil.copy2(temporary, pdf)
record = dict(status='completed', previous_pages=old_count, published_pages=old_count+1,
    previous_sha256=previous, output_sha256=sha(pdf), source_sha256={str(q.relative_to(ROOT)): sha(q) for q in (Path(__file__), packet)},
    prior_page_text='Every previous page retained identically; public-benchmark headline page inserted after cover',
    peak_rss_kib=peak(), wall_s=time.monotonic()-started, scope='Completed public-benchmark results; bounded document rendering, no model fitting/inference.')
(ROOT/'report/publication_public_wins_20261007_v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
