"""Bounded P0-5 refresh: one scoreboard page, preserve all other report pages.

Standard-library collection; ReportLab/PyMuPDF render only. No model, NumPy,
Torch, training or profiling. Input hashes prevent concurrent publication.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import runpy
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/'report/sleeping_machines_status.pdf'
TITLE='Scoreboard — wins, losses and open targets'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect():
    ns=runpy.run_path(str(ROOT/'report/native_language_batched_appendix.py'))
    data=ns['load'](lambda p:json.loads((ns['RES']/p).read_text()))
    public=runpy.run_path(str(ROOT/'report/public_benchmarks_appendix.py'))['load']()
    blocks=runpy.run_path(str(ROOT/'report/scoreboard.py'))['page'](data,public)
    paths={Path(__file__).resolve(),ROOT/'REPORT.md',ROOT/'README.md'}
    paths.update(ROOT/p for p in ('report/scoreboard.py','report/native_language_batched_appendix.py',
        'report/public_benchmarks_appendix.py','report/family_report.py','report/readable_report.py','report/tuned_reference_admission.py',
        'experiments/check_tuned_reference_budgets.py','experiments/e64_lm_baselines.py',
        'experiments/public_benchmarks/collect_neurobench_mg.py','experiments/lm_training_flops.py'))
    paths.update(ROOT/r['path'] for r in [*data['native'],*data['native90'],*data['controls'],*data['controls90']])
    paths.update(ROOT/r['path'] for r in public['mg']['parents'])
    for group, pattern in (('neurobench_primate','curie_pr*.json'),('neurobench_mg','curie_mg*.json')):
        paths.update((ns['RES']/group).glob(pattern))
    paths.update(ns['RES']/p for p in [ns['INFERENCE'],*ns['INFERENCE_MORE'],*ns['SPARSE']])
    paths.update((ROOT/'experiments/queue').glob('aws_tuned_ref_10M_*_20261004T210000Z.txt'))
    for directory in (ns['RES']/'aws_20260929').glob('aws_tuned_ref_10M_*_20261004T210000Z'):
        paths.update(directory.glob('*.json'))
    return blocks,{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def markdown(blocks):
    def text(t): return html.unescape(re.sub(r'<b>(.*?)</b>',r'**\1**',t))
    out=[]
    for kind,value in blocks:
        if kind=='h1': out.append('## '+text(value))
        elif kind=='table':
            header,rows,_=value
            out.append('\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join('---' for _ in header)+' |']
                                 +['| '+' | '.join(text(v) for v in row)+' |' for row in rows]))
        else: out.append(text(value))
    return '\n\n'.join(out)+'\n\n'


def render(stage):
    # Deny numerical libraries even if a rendering dependency probes for them.
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**kw):
        if name.split('.')[0] in ('torch','numpy','h5py','matplotlib'):
            raise ImportError('Numerical imports forbidden in scoreboard rendering')
        return original(name,*a,**kw)
    builtins.__import__=guarded
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
    import pymupdf
    fonts=Path(importlib.util.find_spec('matplotlib').origin).parent/'mpl-data/fonts/ttf'
    for name,file in (('Score','DejaVuSans.ttf'),('ScoreB','DejaVuSans-Bold.ttf')):
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Score',normal='Score',bold='ScoreB',italic='Score',boldItalic='ScoreB')
    blocks=json.loads((stage/'blocks.json').read_text());flow=[]
    for kind,value in blocks:
        if kind=='table':
            header,rows,widths=value
            style=ParagraphStyle('cell',fontName='Score',fontSize=7.4,leading=10)
            table=Table([[Paragraph(html.escape(v),style) for v in row] for row in [header]+rows],
                        colWidths=[w*mm for w in widths],repeatRows=1)
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3fb')),
                ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#1f5fa8')),
                ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe1e8')),
                ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            flow.append(table)
        else:
            style=ParagraphStyle(kind,fontName='ScoreB' if kind=='h1' else 'Score',
                fontSize=14 if kind=='h1' else 8,leading=18 if kind=='h1' else 11,spaceAfter=8)
            flow.append(Paragraph(value,style))
        flow.append(Spacer(1,3))
    doc=SimpleDocTemplate(str(stage/'scoreboard.pdf'),pagesize=A4,leftMargin=16*mm,rightMargin=16*mm,
        topMargin=15*mm,bottomMargin=16*mm,title=TITLE,author='Sleeping Machines project')
    doc.build(flow)
    with pymupdf.open(stage/'previous.pdf') as old,pymupdf.open(stage/'scoreboard.pdf') as score:
        if len(score)!=1: raise ValueError('Scoreboard must fit one page')
        indices=[i for i,p in enumerate(old) if TITLE in p.get_text()]
        if len(indices)!=1: raise ValueError('One existing scoreboard required')
        kept=[i for i in range(len(old)) if i not in indices];merged=pymupdf.open();mapping={}
        merged.insert_pdf(old,from_page=0,to_page=0);mapping[1]=1
        merged.insert_pdf(score);mapping[indices[0]+1]=2
        # Preserve contiguous ranges to avoid duplicating shared PDF objects per page.
        runs=[]
        for i in kept[1:]:
            if not runs or i!=runs[-1][1]+1:runs.append([i,i])
            else:runs[-1][1]=i
        for first,last in runs:
            offset=len(merged);merged.insert_pdf(old,from_page=first,to_page=last)
            for i in range(first,last+1):mapping[i+1]=offset+i-first+1
        inherited=[[level,title,mapping[p]] for level,title,p in old.get_toc() if p in mapping and title!='Scoreboard']
        merged.set_toc([[1,'Scoreboard',2],*inherited]);merged.set_metadata(old.metadata)
        merged.save(stage/'updated.pdf',garbage=4,deflate=True);merged.close()
        with pymupdf.open(stage/'updated.pdf') as new:
            if len(new)!=len(old) or TITLE not in new[1].get_text():raise ValueError('Incorrect scoreboard placement')
            for i in kept:
                if new[mapping[i+1]-1].get_text()!=old[i].get_text():raise ValueError('Changed retained report page')
            for x0,y0,x1,y1,*_ in new[1].get_text('blocks'):
                if min(x0,y0)<0 or x1>new[1].rect.width or y1>new[1].rect.height:raise ValueError('Out-of-bounds scoreboard text')
            validation=dict(total_pages=len(new),scoreboard_page=2,retained_pages=len(kept),
                            retained_page_text='identical on every retained page',numerical_imports=0)
    (stage/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')


def available():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))


def limits():
    os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(1_000_000*1024,1_000_000*1024))


def publish(tag):
    if not tag or Path(tag).name!=tag:raise ValueError('Plain unique tag required')
    stage=ROOT/'.git/scoreboard-preview'/tag;record=ROOT/'report'/f'publication_{tag}.json'
    if stage.exists() or record.exists():raise ValueError('Preserve prior preview/publication')
    if available()<8192*1024:raise ValueError('8 GiB available required')
    blocks,bindings=collect();previous=sha(PDF);stage.mkdir(parents=True)
    (stage/'blocks.json').write_text(json.dumps(blocks,indent=2)+'\n');shutil.copy2(PDF,stage/'previous.pdf')
    started=time.monotonic();peak=0
    with (stage/'render.log').open('x') as log:
        child=subprocess.Popen([sys.executable,__file__,'--render-stage',str(stage)],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),preexec_fn=limits)
        try:
            while child.poll() is None:
                try:rss=int(next(l.split()[1] for l in Path(f'/proc/{child.pid}/status').read_text().splitlines() if l.startswith('VmRSS:')))
                except (FileNotFoundError,StopIteration):rss=0
                peak=max(peak,rss)
                if rss>300000 or available()<8192*1024 or time.monotonic()-started>120:raise ValueError('Publication resource guard tripped')
                time.sleep(.1)
            if child.returncode:raise ValueError('Render failed: '+str(stage/'render.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    if sha(PDF)!=previous or any(sha(ROOT/p)!=h for p,h in bindings.items()):raise ValueError('Concurrent report/input change')
    text=(ROOT/'REPORT.md').read_text();heading='## '+TITLE
    start=text.index(heading);end=text.find('\n## ',start+len(heading))
    if end<0:raise ValueError('Scoreboard section boundary missing')
    text=text[:start]+text[end+1:];boundary=text.index('## ')
    report_text=text[:boundary]+markdown(blocks)+text[boundary:]
    readme=(ROOT/'README.md').read_text();start,end='<!-- scoreboard:start -->','<!-- scoreboard:end -->'
    block=start+'\n'+markdown(blocks)+end
    readme_text=readme[:readme.index(start)]+block+readme[readme.index(end)+len(end):]
    archive=ROOT/'report/archive'/tag;archive.mkdir()
    shutil.copy2(PDF,archive/'previous.pdf');shutil.copy2(ROOT/'REPORT.md',archive/'REPORT.md');shutil.copy2(ROOT/'README.md',archive/'README.md')
    shutil.copy2(stage/'updated.pdf',PDF.with_suffix('.publishing.pdf'));PDF.with_suffix('.publishing.pdf').replace(PDF)
    (ROOT/'REPORT.md').write_text(report_text);(ROOT/'README.md').write_text(readme_text)
    validation=json.loads((stage/'validation.json').read_text())
    record.write_text(json.dumps(dict(status='completed',published_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=bindings,previous_pdf_sha256=previous,
        outputs={p:sha(ROOT/p) for p in ('REPORT.md','README.md','report/sleeping_machines_status.pdf')},
        validation=validation,peak_render_rss_kib=peak,wall_s=time.monotonic()-started,
        guards=dict(threads=1,rss_kib=300000,address_space_kib=1000000,min_available_mib=8192,timeout_s=120),
        scope='One corrected scoreboard page; all other report pages retained verbatim, including historical protocol assertions. '
              'New scoreboard is the explicit context correction. No training/model runtime or new benchmark score.'),indent=2)+'\n')
    print(json.dumps(dict(publication=str(record.relative_to(ROOT)),peak_rss_kib=peak,**validation)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tag');mode.add_argument('--render-stage');args=parser.parse_args()
    if args.tag:publish(args.tag)
    else:render(Path(args.render_stage))
