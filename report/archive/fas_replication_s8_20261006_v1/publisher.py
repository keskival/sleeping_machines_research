import sys,os,resource,time,json,hashlib,shutil,runpy,html
from pathlib import Path
root=Path('/workspace');sys.path.insert(0,str(root));start=time.monotonic()
resource.setrlimit(resource.RLIMIT_AS,(1000000*1024,1000000*1024))
def available():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert available()>=8192*1024
blocks=runpy.run_path(str(root/'report/fas_native_replication.py'))['pages']()[0]
pdf=root/'report/sleeping_machines_status.pdf';md=root/'REPORT.md';prev=sha(pdf)
folder=root/'report/archive/fas_replication_s8_20261006_v1';folder.mkdir(exist_ok=False)
shutil.copy2(pdf,folder/'previous.pdf');shutil.copy2(md,folder/'REPORT.md')
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer
from reportlab.lib import colors
import pymupdf
flow=[];markdown=[]
for kind,value in blocks:
 if kind=='table':
  headers,rows,widths=value;style=ParagraphStyle('cell',fontSize=9,leading=13)
  table=Table([[Paragraph(html.escape(v),style) for v in row] for row in [headers]+rows],colWidths=[w*mm for w in widths]);table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3fb')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flow.append(table)
  markdown.append('\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(r)+' |' for r in rows]))
 else:
  style=ParagraphStyle(kind,fontName='Helvetica-Bold' if kind=='h1' else 'Helvetica',fontSize=16 if kind=='h1' else 10,leading=21 if kind=='h1' else 15)
  flow.append(Paragraph(value,style));markdown.append(('## ' if kind=='h1' else '')+value.replace('<b>','**').replace('</b>','**'))
 flow.append(Spacer(1,12))
page=folder/'replication_page.pdf';SimpleDocTemplate(str(page),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm).build(flow)
with pymupdf.open(pdf) as old,pymupdf.open(page) as add:
 assert len(add)==1 and len(add[0].get_text())>500
 old_count=len(old)
 existing=[i for i,page in enumerate(old) if 'FAS native frozen-recipe replication' in page.get_text()]
 assert existing==[old_count-1]
 old.delete_page(existing[0]);old.insert_pdf(add)
 temporary=pdf.with_suffix('.replication-building.pdf');old.save(temporary,garbage=0,deflate=False)
assert sha(pdf)==prev and available()>=8192*1024
process_peak=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')))
assert process_peak<300000
with pymupdf.open(temporary) as new,pymupdf.open(folder/'previous.pdf') as previous:
 assert len(new)==len(previous)
 assert all(new[i].get_text()==previous[i].get_text() for i in range(len(previous)-1))
 new[-1].get_pixmap().save('/tmp/fas_replication_page.png')
process_peak=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')))
assert process_peak<300000 and available()>=8192*1024
temporary.replace(pdf)
text=md.read_text();cut=text.rfind('\n## FAS native frozen-recipe replication');assert cut>=0
md.write_text(text[:cut]+'\n'+'\n\n'.join(markdown)+'\n')
record=dict(status='completed',previous_pdf_sha256=prev,pdf_sha256=sha(pdf),previous_pages=old_count,published_pages=old_count,retained_page_text='identical on every retained page',peak_rss_kib=process_peak,wall_s=time.monotonic()-start,source_sha256={'report/fas_native_replication.py':sha(root/'report/fas_native_replication.py')},guards=dict(min_available_mib=8192,rss_kib=300000,address_space_kib=1000000),scope='Completed FAS artifact evidence only; no training or model inference')
(root/'report/publication_fas_replication_s8_20261006_v1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
