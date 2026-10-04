"""Bounded, vector-only model-family chapter; preserve the existing report.

No model, NumPy, Torch, plotting or training imports. The canonical report
builder also calls integrate() so subsequent evidence rebuilds retain this view.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'report/sleeping_machines_status.pdf'
MARKER = 'Sleeping Machines | model family chapter'
LEGACY_MARKER = 'SM-FAMILY-CHAPTER'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_frontmatter():
    return '''## The model family and its place in the landscape — 4 October 2026

**Sleeping Machines is a family of causal networks of stateful temporal
programs.** Content, computational time, addresses and persistent evidence
jointly determine computation. The shared interface supports different local
programs and heterogeneous compositions; today's fitted native model is one
member. The integrated target combines computational delays/races, hard
selective state updates, small learned messages, separate keys/values and
useful credit to unrealized alternatives, with capacity beyond costly activity.

![Position among model families](report/figures/architecture_landscape.svg)

The family intersects recurrence/SSMs, attention/Transformers, sparse experts,
statistical and historical memory, TTT and event/temporal systems. Exact
containment needs their actual operators, information, state and schedule.
The contribution sought is their coherent temporal/selective integration and
its quality/learning/resource behavior, not a claim that familiar primitives
are new or that one member already matches every incumbent.

Selective and dense regions can coexist. Synchronous barriers are permitted
local schedules; the general semantics require no global periodic update tick.
Rich interaction can be added where a task requires it, while other regions
remain economical. Online learning and batching use the same structure.
Native gradient TTT, general structural adaptation, shared-world multimodal
learning and clockless hardware are capabilities/ambitions with their own
integration and measurement requirements.

**Read in order:** [overview](report/model_family_overview.md) →
[formal core](report/model_family_specification.md) →
[complete contrasting members](report/model_family_members.md) →
[design choices](report/model_family_design.md) and
[composition rules](report/model_family_composition.md).
The [visual atlas](report/architecture_atlas.html),
[implementation review](report/architecture_review.md) and
[claim/evidence map](report/architecture_evidence.md) supply detail and scope.
Computational power/work, representation and trainability are reviewed from
atoms to complete systems. The illustrative members are specifications, not
new benchmark results. Forward preservation, learning consistency and resource
improvement are separate contracts.

'''


def sections():
    fields = [
        ('E / Evidence','Observation schema, provenance, time/address adapters and admissible histories'),
        ('O / Operators','Actual maps, domains, precision and rounding for numerical claims'),
        ('G / Composition','Units, connections, shared paths and structural-action space'),
        ('S / State','Facts, timing, queues, caches, randomness, ownership and resets'),
        ('I / Interfaces','Content/address/time types, conversions and state access'),
        ('P / Participation','Discovery, keys, proposals, deliveries, writes and alternative support'),
        ('T / Schedule','Order, delays, ties, deadlines, joins, cancellation and barriers'),
        ('Q / Queries','Observed cutoff, accessible snapshot, completion and readout'),
        ('L / Objectives','Task/self-supervision, labels, scoring order and resource penalties'),
        ('U / Learner','Credit, horizon, optimizer/update ownership and versions'),
        ('B / Bounds','Finite execution, capacities, admission and overflow policy'),
        ('X / Execution','Reference/deployment equivalence, measurement and evidence status')]
    return [
        dict(title='The model family and the landscape',figure='landscape',
             before=['Sleeping Machines is a family of causal networks of stateful temporal programs. Content, computational time, addresses and persistent evidence jointly determine computation. A common interface supports different programs and mixtures at every level.'],
             after=['The envelope supplies choices; a member fixes compatible operators and contracts; the integrated target tests temporal/selective quality and resource behavior. Conditional inclusion is distinct from learned parity or lower cost.']),
        dict(title='A generative design space, from atoms to systems',figure='design_space',
             before=['Choose evidence, flow, writes, keys, reception, output and credit; then compose modules, interactions, layers, stacks and complete learning systems. Uniform width, fan-in and scheduling are optional design choices.'],
             after=['Review computational power and total work, represented distinctions, and useful learning at each level. Expand the earliest limiting region: downstream richness cannot recover erased evidence.']),
        dict(title='The same three axes at every level',
             before=['Computational power and required work, represented distinctions, and useful learning are separate properties. Each must survive the interfaces that compose a model. More expressivity alone does not establish easier optimization or generalization.'],
             table=(['Level','Power / work','Representation','Trainability'],[
                 ['Atoms','Flow, delays, gates, races, sums and their ordering','Content, order, interval, phase and repeated evidence','Continuous sensitivity plus event-boundary utility'],
                 ['Units','State transformation, write, emission and readout','Retained facts, temporal modes and message bandwidth','Content/time/state credit; conditioning and exposure'],
                 ['Modules','Discovery, candidate proposals, reception and commits','Accessible evidence and which messages can meet','Supported alternatives, selection and write utility'],
                 ['Layers','Mixing, shared reads, transport and timed joins','Relations across programs, sources and values','Coupled derivatives and changed event histories'],
                 ['Stacks','Depth, recurrence and optional continuation','Hierarchical features and persistent context','Information paths, Jacobian products and writer horizon'],
                 ['Compositions','Neural, statistical, historical and rich local regions','Abstraction plus direct evidence at a common query','Fusion responsibilities, retrieval and future utility'],
                 ['Systems','Schedule, caches, batching, optimizer and serving','Usable state within information/resource limits','Causal objectives, versions, update consistency and cost']]),
             table_widths=[77,143,143,144.276],
             after=['Diagnose lost information, unsupported computation and missing credit separately. Then test their coupled consequence in an integrated member, preserving the successful parent and the full resource boundary.']),
        dict(title='Relationships to established model families',
             before=['The envelope intersects existing categories and permits conditional reference constructions. Its proposed contribution is an economical integration of time, state, selection and learning; existing systems can also be extended or hybridized.'],
             table=(['Family','What can be included','Additional design freedom','Condition / price'],[
                 ['Recurrence / SSM','Supported state transitions and filters','Addressed private programs and timed nonlinear interaction','Actual operators, retained state and stable learning'],
                 ['Attention / Transformer','Supported attention and full reference blocks','Persistent evidence, races and variable delivery fan-in','Full aggregation, FFN/residual/norm/positions; all required work'],
                 ['Sparse experts / MoE','Supported gating and expert mixtures','Private stateful experts, clocks and future write utility','Candidate discovery, mixture aggregation and trained specialization'],
                 ['Counts / retrieval','Causal estimators and explicit historical evidence','Shared learned abstractions and temporal/nonlocal fusion','Correct statistics, support, access, updates and storage'],
                 ['Adaptive memory / TTT','Specified inner learner and causal deployment updates','Selective temporal programs and local learning policies','Actual objective/update, versions, retained credit and extra work'],
                 ['Event / temporal','Local event scheduling, timers and supported temporal logic','Small vector messages and coupled alternative-route credit','Precision, ordering, termination, traffic and measured hardware']]),
             table_widths=[88,139,143,137.276],
             after=['A function inclusion is not a trained-quality or resource result. A restricted winner-only member has less interaction support than a full-aggregation member. Broader operator choices provide a route to matching needed functions without promising that present native units already implement every reference.']),
        dict(title='Units combine content, timing and persistent evidence',figure='receiver',
             before=['The native receiver below illustrates one branch. It combines a stored key, input-conditioned temporal proposal, small residual value and computational clock. Keys choose access; values carry content; the selected write changes future evidence.'],
             after=['A unit is a product of choices: statistical fields, neural vectors, protected facts or fast parameters; identity, decay, rotation or richer local maps; first, multi-message, window or silence reception. Their integration changes the function and the credit needed to learn it.']),
        dict(title='Modules, interactions, layers and stacks',figure='stack',
             before=['Candidate discovery, winner selection, reception and continuation are distinct decisions. Heads expose complementary evidence only through declared mixing, joins or shared reads. The diagram is the native branch, not the whole family definition.'],
             after=['Depth creates hierarchical temporal functions when information and credit survive interfaces. Persistent facts, accessible facts and trainable writers are separate capacities. Shared maps do not automatically share private state across sources.']),
        dict(title='One family across dense and sparse worlds',figure='shared_world',
             before=['Dense frames, language, queries and irregular events can feed explicit shared information paths. Input density, computation density, arrival cadence and physical clocking are independent choices.'],
             after=['A rich local island can reproduce supported attention or other reference operators while surrounding regions remain selective. Full participation needs actual aggregation; synchrony needs snapshots/barriers. Joint multimodal training remains an integration ambition. No global periodic tick is required; physical clockless energy is unmeasured.']),
        dict(title='The same structure can learn, batch and adapt',figure='learning',
             before=['Representation, timing, routes and writes form one coupled credit problem. Factual derivatives teach a realized history; useful alternative returns teach changed routes, reception and later state consequences. The successful local value teacher does not cover full future-write utility.'],
             after=['An earlier neural family pilot improved causal online prediction at 10.736 times total processing work. It is scoped evidence, not current native-backbone TTT or drift retention. State/statistical adaptation is distinct from gradient TTT. Local, streamed and batched updates need ownership and versions. General morphing and fully asynchronous hardware training remain open; learned event routes already exist.']),
        dict(title='A precise core with contrasting complete members',table=(['Field','Required contract'],fields),
             before=['A member declares the following twelve semantic fields. A construction grammar permits serial, parallel, recurrent, memory-linked, scheduled and policy-controlled regions, subject to compatibility and termination.'],
             after=['C: causal binary counts, addressed increments and a full prediction distribution; no gradient TTT. R: two private temporal programs, hard race and a local online value teacher; no deep or full writer credit. H: shared sparse reception, two-record dense query and exact finite-option episode TTT. These complete illustrative specifications share one template; they are not new fitted models or benchmark claims.']),
        dict(title='Composition and adaptive growth have separate contracts',figure='composition',
             before=['Local capability reaches the system only through compatible information, state, time, query, credit and resource interfaces. Serial, parallel, recurrent and hybrid compositions expose different evidence and dependencies.'],
             after=['Forward preservation checks future state and emissions, including pending deadlines. Learning preservation additionally checks credit and parameter/optimizer migration. Resource refinement checks actual work, storage, traffic and latency. Equal current predictions do not establish any of these stronger claims. Intended learned deviations remain permitted.']),
        dict(title='Evidence supports the family; integration sets the agenda',evidence=True,
             before=['Completed native results support content-bearing temporal computation, useful route credit and capacity beyond selected writes. The rows below share T256 and the same restricted 1M test region; each is single-seed. The 90M row changes data and width.'],
             after=['Value choice credit improves the matched 10M D4/P32/U2 result by 0.134895 BPC for about 0.30% extra estimated fitting work. U2→U4 adds useful capacity with the same eight writes, but increases fitting work and untied parameters. These support scoped mechanisms; they are not modern frontier or public full-test wins.',
                    'Next integrated proof points: complete future-write utility and useful horizons; discovery/reception/silence; trained sparse-state and quality parity; matched repeated controls and measured traffic/energy. Native TTT, automatic structural design and shared-world integration require their own evidence.',
                    'The investment thesis is a reusable model/runtime/compute substrate from datacenters to mobile and robotics. The family definition makes that breadth concrete; commercial advantage remains an empirical quality/resource and adoption question.'])]


def build_chapter(output):
    import pymupdf
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    font_dir = Path(importlib.util.find_spec('matplotlib').origin).parent/'mpl-data/fonts/ttf'
    for name, file in [('Family','DejaVuSans.ttf'),('FamilyB','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
    width,height = 595.276,841.89
    margin, body_width = 44.,507.276
    base = output.with_suffix('.text.pdf')
    c = canvas.Canvas(str(base),pagesize=(width,height),pageCompression=1)
    c.setTitle('Sleeping Machines — model family and landscape')
    c.setAuthor('Tero Keski-Valkama and Karoliina Salminen')
    overlays=[]
    for index,section in enumerate(sections()):
        y=48.
        def paragraph(value,size=10.5,bold=False,color='#25354b'):
            nonlocal y
            p=Paragraph(html.escape(value),ParagraphStyle('family',fontName='FamilyB' if bold else 'Family',
                fontSize=size,leading=size*1.42,textColor=HexColor(color)))
            _,h=p.wrap(body_width,1000)
            if y+h>height-64: raise ValueError('Family page overflow: '+section['title'])
            p.drawOn(c,margin,height-y-h);y+=h+12
        paragraph(section['title'],17,True,'#2259a7')
        for text in section['before']: paragraph(text)
        if section.get('figure'):
            svg=ROOT/'report/figures'/('architecture_'+section['figure']+'.svg')
            with pymupdf.open(stream=svg.read_bytes(),filetype='svg') as source:
                svg_pdf=source.convert_to_pdf()
            with pymupdf.open(stream=svg_pdf,filetype='pdf') as drawing:
                ratio=drawing[0].rect.height/drawing[0].rect.width
            w=body_width;h=w*ratio
            if h>455: h=455;w=h/ratio
            if y+h>height-120: raise ValueError('Family diagram overflow')
            overlays.append((index,(margin+(body_width-w)/2,y,margin+(body_width+w)/2,y+h),svg_pdf))
            y+=h+15
        data=section.get('table')
        if section.get('evidence'):
            evidence=json.loads((ROOT/'report/architecture_evidence.json').read_text())
            data=(['Native member','Fit M','BPC','Fit TFLOPs','MFLOPs/target'],[
                [s['label'].replace('D4 / ','').replace(', value choice credit',', credit'),
                 f"{s['fitting_presentations']/1e6:.3f}",f"{s['test_bpc']:.4f}",
                 f"{s['estimated_whole_fit_tflops']:.3f}",f"{s['estimated_fit_mflops_per_presentation']:.3f}"]
                for s in evidence['snapshots']])
        if data:
            headers,rows=data
            style=ParagraphStyle('cell',fontName='Family',fontSize=8.6,leading=12)
            cells=[[Paragraph(html.escape(str(v)),style) for v in row] for row in [headers]+rows]
            widths=section.get('table_widths') or ([115,body_width-115] if len(headers)==2 else [187,55,60,93,body_width-395])
            table=Table(cells,colWidths=widths)
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),HexColor('#eaf0f9')),
                ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
                ('LINEBELOW',(0,0),(-1,-1),.3,HexColor('#d4dfeb'))]))
            _,h=table.wrap(body_width,1000)
            if y+h>height-80: raise ValueError('Family table overflow')
            table.drawOn(c,margin,height-y-h);y+=h+17
            if section.get('evidence'):
                paragraph('Saved arithmetic plus unit-counted special functions; eager full-step estimates. Whole-fit and per-fitting-target columns have consistent units across rows. Evaluation, setup, traffic and energy are additional.',9)
        for text in section['after']: paragraph(text)
        c.setFont('Family',7);c.setFillColor(HexColor('#53657d'))
        c.drawString(margin,31,f'{MARKER} | 4 October 2026 | private research report')
        c.drawRightString(width-margin,31,f'Family {index+1} / {len(sections())}')
        c.showPage()
    c.save()
    with pymupdf.open(base) as doc:
        for index,rect,pdf in overlays:
            with pymupdf.open(stream=pdf,filetype='pdf') as svg_doc:
                doc[index].show_pdf_page(pymupdf.Rect(rect),svg_doc,0)
        doc.set_toc([[1,s['title'],i+1] for i,s in enumerate(sections())])
        doc.save(output,garbage=3,deflate=True)
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules


def integrate(source, destination, chapter):
    """Replace only marker-labelled family pages; retain every other page."""
    import pymupdf
    with pymupdf.open(source) as old,pymupdf.open(chapter) as family:
        kept=[i for i,p in enumerate(old) if not any(marker in p.get_text() for marker in (MARKER,LEGACY_MARKER))]
        if not kept: raise ValueError('Report has no retained evidence pages')
        merged=pymupdf.open(); mapping={}
        merged.insert_pdf(old,from_page=kept[0],to_page=kept[0]);mapping[kept[0]+1]=1
        merged.insert_pdf(family)
        runs=[]
        for i in kept[1:]:
            if not runs or i!=runs[-1][1]+1: runs.append([i,i])
            else: runs[-1][1]=i
        for first,last in runs:
            offset=len(merged)
            merged.insert_pdf(old,from_page=first,to_page=last)
            for i in range(first,last+1): mapping[i+1]=offset+i-first+1
        toc=[[1,'Model family and landscape',2]]+[[2,s['title'],i+2] for i,s in enumerate(sections())]
        inherited=[[level,title,mapping[page]] for level,title,page in old.get_toc() if page in mapping]
        # Existing reports may have their own hierarchy; keep it under a root.
        if inherited:
            toc.append([1,'Preserved research report',1])
            toc.extend([[level+1,title,page] for level,title,page in inherited])
        merged.set_toc(toc)
        merged.set_metadata(dict(old.metadata,title='Sleeping Machines — model family and research status'))
        merged.save(destination,garbage=4,deflate=True)
        merged.close()
    with pymupdf.open(source) as old,pymupdf.open(destination) as new:
        assert len(new)==len(kept)+len(sections())
        for i in kept:
            if old[i].get_text()!=new[mapping[i+1]-1].get_text(): raise ValueError('Changed evidence page')
        for i in range(1,1+len(sections())):
            if MARKER not in new[i].get_text(): raise ValueError('Missing family marker')
            for x0,y0,x1,y1,*_ in new[i].get_text('blocks'):
                if min(x0,y0)<0 or x1>new[i].rect.width or y1>new[i].rect.height:
                    raise ValueError('Out-of-bounds family text')
    return dict(retained_pages=len(kept),family_pages=len(sections()),total_pages=len(kept)+len(sections()),
                retained_evidence_text='identical on every retained page')


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))


def limits():
    os.nice(19)
    resource.setrlimit(resource.RLIMIT_AS,(1_000_000*1024,1_000_000*1024))


def publish(tag):
    if not tag or Path(tag).name!=tag: raise ValueError('Fresh plain tag required')
    stage=ROOT/'.git/family-report-preview'/tag
    record=ROOT/'report'/('publication_'+tag+'.json')
    if stage.exists() or record.exists(): raise ValueError('Preserve prior stage/record')
    if available()<8192*1024: raise ValueError('Less than 8 GiB available')
    paths=[Path(__file__).resolve(),ROOT/'REPORT.md',ROOT/'report/architecture_evidence.json']
    paths+=sorted((ROOT/'report').glob('model_family_*.md'))+sorted((ROOT/'report/figures').glob('architecture_*.svg'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};previous=sha(OUTPUT)
    stage.mkdir(parents=True);shutil.copy2(OUTPUT,stage/'previous.pdf')
    start=time.monotonic();peak=0
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (stage/'render.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--render-stage',str(stage)],
            stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=limits)
        try:
            while child.poll() is None:
                try: rss=next((int(x.split()[1]) for x in Path(f'/proc/{child.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')),0)
                except FileNotFoundError: rss=0
                peak=max(peak,rss)
                if rss>300000 or available()<8192*1024 or time.monotonic()-start>120: raise RuntimeError('Report resource guard tripped')
                time.sleep(.1)
            if child.returncode: raise RuntimeError('Family render failed; inspect '+str(stage/'render.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: child.kill();child.wait()
    if previous!=sha(OUTPUT) or hashes!={str(p.relative_to(ROOT)):sha(p) for p in paths}: raise ValueError('Concurrent report/source change')
    archive=ROOT/'report/archive'/(tag+'_previous.pdf')
    if archive.exists(): raise ValueError('Preserve previous report archive')
    shutil.copy2(OUTPUT,archive)
    temporary=OUTPUT.with_suffix('.publishing.pdf');shutil.copy2(stage/OUTPUT.name,temporary);temporary.replace(OUTPUT)
    result=json.loads((stage/'validation.json').read_text())
    record.write_text(json.dumps(dict(status='completed',published_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes,previous_sha256=previous,output_sha256=sha(OUTPUT),validation=result,
        guards=dict(threads=1,rss_kib=300000,address_space_kib=1000000,min_available_mib=8192,timeout_s=120),
        render_wall_s=time.monotonic()-start,peak_render_rss_kib=peak,
        scope='Vector family chapter and preserved report pages; no numerical model or new benchmark'),indent=2)+'\n')
    print(json.dumps(dict(pdf=str(OUTPUT),preview=str(stage),peak_rss_kib=peak,**result)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--tag');mode.add_argument('--render-stage')
    args=parser.parse_args()
    if args.tag: publish(args.tag)
    else:
        stage=Path(args.render_stage);build_chapter(stage/'family.pdf')
        result=integrate(stage/'previous.pdf',stage/OUTPUT.name,stage/'family.pdf')
        (stage/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
