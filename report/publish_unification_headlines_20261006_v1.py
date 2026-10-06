"""Insert completed cross-domain headlines after the cover; retain all evidence."""
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
packet = ROOT/'report/unification_headline_evidence_20261006_v1.json'
e = json.loads(packet.read_text())
assert e['status'] == 'completed'
for source, digest in e['input_sha256'].items(): assert sha(ROOT/source) == digest, source
folder = ROOT/'report/archive/unification_headlines_20261006_v1'
pdf = ROOT/'report/sleeping_machines_status.pdf'
previous = sha(pdf)
assert not (folder/'previous.pdf').exists()
shutil.copy2(pdf, folder/'previous.pdf')
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
import pymupdf
body = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=12)
cell = ParagraphStyle('cell', fontName='Helvetica', fontSize=8, leading=11)
heading = ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=18, leading=23)
def p(text, style=body): return Paragraph(html.escape(text), style)
f256, f512 = e['fas']['256'], e['fas']['512']
language = e['language']
rows = [
    ['Evidence front', 'Completed indication', 'Scope and next proof'],
    ['Anonymous interleaved processes',
     f"FAS mean AUROC {f256['mean']:.4f} vs {f256['best_six_generic_controls']:.4f} at 256 events; {f512['mean']:.4f} vs {f512['best_six_generic_controls']:.4f} at 512 events. Every native seed beats the best generic control at both points.",
     'Three native seeds; FAS v1 quality wins against six saved anonymous generic controls. Privileged diagnostics excluded. Sealed v2 and stronger neural references follow.'],
    ['Mixed-type tables, small data',
     '100% on 256 synthetic DEV rows from 64 FIT rows; 6,370 learned parameters. Typed numeric, categorical, Boolean and missingness comparisons feed the integrated temporal core.',
     'One seed, fixed predicates. Relabeling, column order, missing-value semantics and actual alternative-write credit checks pass. Learned discovery and real tables versus trees follow.'],
    ['Token language and useful memory',
     f"Two seeds at 1M TRAIN tokens: {language['selected_dev_nll'][0]:.4f} / {language['selected_dev_nll'][1]:.4f} DEV NLL, mean {language['mean']:.4f}. Memory erasure raises loss {language['memory_erasure_nll_deltas'][0]:.4f} / {language['memory_erasure_nll_deltas'][1]:.4f}.",
     'GPT-2 FineWeb, two passes, 2,040 DEV targets. Public Transformer comparison, complete larger fitting work and reserved 4M point follow. No fitted scaling law.'],
]
flow = [p('One substrate: three completed evidence anchors', heading), Spacer(1,12),
    p('The ambition spans language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. The evidence now connects token sequences, anonymous interleaved process logs and mixed-type comparisons within the temporal/sparse family.'), Spacer(1,12)]
t = Table([[p(value, cell) for value in row] for row in rows], colWidths=[105,205,180], repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3fb')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
flow += [t, Spacer(1,14), p('The reusable construction is the advantage: type-respecting interfaces produce addressed events; persistent memory, temporal computation, selective updates and counterfactual credit learn their consequences. Capacity, selected activity, timing and learning are separate design axes.'), Spacer(1,10),
    p('These instances were fitted separately. The next unification test shares trained core parameters across domains and measures transferred skills and retention. Source-bound headline packet: report/unification_headline_evidence_20261006_v1.json. All historical evidence pages are retained.')]
appendix = folder/'headline_page.pdf'
SimpleDocTemplate(str(appendix), pagesize=A4, leftMargin=45, rightMargin=45, topMargin=40, bottomMargin=40).build(flow)
temporary = folder/'published.pdf'
with pymupdf.open(pdf) as old, pymupdf.open(appendix) as add:
    old_count = len(old)
    assert len(add) == 1
    old.insert_pdf(add, start_at=1)
    old.save(temporary, garbage=0, deflate=False)
with pymupdf.open(temporary) as new, pymupdf.open(folder/'previous.pdf') as old:
    assert len(new) == len(old)+1
    for i in range(len(old)): assert new[i if i == 0 else i+1].get_text() == old[i].get_text()
    text = new[1].get_text()
    assert all(value in text for value in ('0.5924','0.5587','100%','7.2575','fitted separately'))
    for x0,y0,x1,y1,*_ in new[1].get_text('blocks'):
        assert min(x0,y0) >= 0 and x1 <= new[1].rect.width and y1 <= new[1].rect.height
    new[1].get_pixmap().save(str(folder/'preview.png'))
assert peak() < 300000 and available() >= 8192*1024 and sha(pdf) == previous
shutil.copy2(temporary, pdf)
record = dict(status='completed', previous_pages=old_count, published_pages=old_count+1,
    previous_sha256=previous, output_sha256=sha(pdf), source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),packet)},
    prior_page_text='Every previous page retained identically; headline page inserted after cover',
    peak_rss_kib=peak(), wall_s=time.monotonic()-started, scope='Completed cross-domain evidence; bounded document rendering, no model fitting/inference.')
(ROOT/'report/publication_unification_headlines_20261006_v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
