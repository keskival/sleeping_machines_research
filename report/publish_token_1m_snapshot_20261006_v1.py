"""Append completed 1M evidence, preserving all previous PDF pages."""
import hashlib, html, json, resource, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
resource.setrlimit(resource.RLIMIT_AS,(1000000*1024,1000000*1024))
start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def available():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))
assert available()>=8192*1024
from report.token_data_growth import measured_pages,stage_pages
blocks=measured_pages()+stage_pages(1048576,'1m')
assert len(blocks)==2
utility=json.loads((ROOT/'experiments/results/diagnostics/curie_original_1m_p24_s7_utility_20261006_v1.json').read_text())
assert utility['status']=='completed' and utility['rows'][0]['matched_rng'] and utility['rows'][0]['partition_parity']
blocks[-1].append(('p','Independent P24 seed7 selects7.263557 at finalstep4,096; seed6 selects7.251503 atstep2,048, mean7.257530NLL. Frozen seed7 context gain0.995919; memory erasure+0.017380, full-message-state erasure+0.475144, both+0.443788. Two passes and every optimizer update remain charged. These are development results; public Transformer quality and equal complete fitting-compute comparison are the subsequent benchmark tests.'))
folder=ROOT/'report/archive/token_1m_snapshot_20261006_v1'
folder.mkdir(exist_ok=False)
pdf=ROOT/'report/sleeping_machines_status.pdf';previous=sha(pdf)
shutil.copy2(pdf,folder/'previous.pdf');shutil.copy2(ROOT/'REPORT.md',folder/'REPORT.md')
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,Image,PageBreak
from reportlab.lib import colors
import pymupdf
style=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=12)
cell=ParagraphStyle('cell',fontName='Helvetica',fontSize=7,leading=9)
heading=ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=15,leading=19)
flow=[]
for index,page in enumerate(blocks):
 if index:flow.append(PageBreak())
 for kind,value in page:
  if kind=='table':
   headers,rows,widths=value
   scale=490/sum(widths)
   table=Table([[Paragraph(html.escape(str(v)),cell) for v in row] for row in [headers]+rows],colWidths=[w*scale for w in widths],repeatRows=1)
   table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3fb')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));flow.append(table)
  elif kind=='figure':
   name,_=value;flow.append(Image(str(ROOT/f'report/figures/{name}.png'),width=490,height=490*4.1/12))
  else:flow.append(Paragraph(html.escape(value),heading if kind=='h1' else style))
  flow.append(Spacer(1,8))
appendix=folder/'appendix.pdf'
SimpleDocTemplate(str(appendix),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=40).build(flow)
temporary=folder/'published.pdf'
with pymupdf.open(pdf) as old,pymupdf.open(appendix) as add:
 old_count=len(old);added=len(add)
 assert added>=2 and all(len(p.get_text())>100 for p in add)
 old.insert_pdf(add);old.save(temporary,garbage=0,deflate=False)
with pymupdf.open(temporary) as new,pymupdf.open(folder/'previous.pdf') as old:
 assert len(new)==len(old)+added
 assert all(new[i].get_text()==old[i].get_text() for i in range(len(old)))
 text='\n'.join(new[i].get_text() for i in range(len(old),len(new)))
 assert all(x in text for x in ('7.251503','7.263761','0.995919','7.257530'))
 for i in range(added):new[old_count+i].get_pixmap().save(str(folder/f'preview_{i}.png'))
assert peak()<300000 and available()>=8192*1024 and sha(pdf)==previous
shutil.copy2(temporary,pdf)
record=dict(status='completed',previous_pdf_sha256=previous,pdf_sha256=sha(pdf),previous_pages=old_count,published_pages=old_count+added,retained_page_text='identical on every previous page',peak_rss_kib=peak(),wall_s=time.monotonic()-start,scope='Completed 1M development evidence; no training or model inference; all historical PDF pages preserved',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'report/token_data_growth.py',ROOT/'report/figures/token_language_measured_scaling_20261006_v4.json']})
(ROOT/'report/publication_token_1m_snapshot_20261006_v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
