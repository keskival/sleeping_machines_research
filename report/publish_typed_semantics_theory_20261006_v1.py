"""Bounded document publication of typed theory; retain every evidence page."""
import hashlib, html, json, resource, shutil, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
resource.setrlimit(resource.RLIMIT_AS,(1000000*1024,1000000*1024))
started=time.monotonic()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def available(): return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
def peak(): return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))
assert available()>=8192*1024
receipt=ROOT/'experiments/results/diagnostics/typed_semantics_composition_contracts_20261006_v1.json'
r=json.loads(receipt.read_text());assert r['status']=='completed' and r['cases']==39
for source,digest in r['source_sha256'].items(): assert sha(ROOT/source)==digest,source
folder=ROOT/'report/archive/typed_semantics_theory_20261006_v1'
pdf=ROOT/'report/sleeping_machines_status.pdf';previous=sha(pdf)
assert not (folder/'previous.pdf').exists();shutil.copy2(pdf,folder/'previous.pdf')
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
import pymupdf
body=ParagraphStyle('body',fontName='Helvetica',fontSize=9,leading=12)
title=ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=17,leading=22)
sub=ParagraphStyle('sub',fontName='Helvetica-Bold',fontSize=11,leading=14)
blocks=[
 ('title','Typed learning: preserve meaning, then compose evidence'),
 ('body','The platform case combines type-respecting interfaces, conditional decisions and deep temporal state. Numeric thresholds, category membership, Boolean tests and missingness produce meaningful events before learned representations mix. This connects heterogeneous tables with the same addressed, persistent event construction used for language and asynchronous experience.'),
 ('sub','A precise objection to raw encoding arithmetic'),
 ('body','A fixed affine gate h(c,z)=w*c+b(z), with nominal category code c and other inputs z fixed, cannot use c and remain invariant under every permutation of its codes: swapping distinct codes forces w=0. A categorical embedding table relabeled with its schema avoids this obstruction. Coefficients can convert physical units; sums in a declared common latent space are legal. The claim concerns semantic assumptions, not a prohibition on neural arithmetic.'),
 ('sub','The typed event composition result'),
 ('body','If a typed adapter produces identical event identities, payloads and computational timing under a valid representation change, the same causal core, initial state and random draws produce identical state trajectories and outputs. Induction over events and depth proves the result. For numeric margin m=(x-t)/s, positive affine changes x\'=a*x+b, t\'=a*t+b, s\'=a*s preserve m and its comparison clocks. Arbitrary monotone changes preserve order cuts, not necessarily margins.'),
 ('body','Equivalent learning additionally requires appropriate preprocessing, initialization, credit and optimizer coordinates. Raw threshold gradients rescale with units; an unchanged raw-unit optimizer step need not preserve training. Inference invariance alone does not prove learning invariance.'),
 ('sub','Race learning retains computation through time'),
 ('body','Rates exp(m), exp(-m) give joint winner/time credit. For loss A_W+B*T, expected risk is p*A_true+(1-p)*A_false+B/Lambda. Its derivative includes both branch utility and the change in first-event time. Equal branch losses can leave nonzero timing credit. Actual alternative suffix evaluations do not eliminate time credit or direct factual derivatives.'),
 ('sub','Evidence and the next experiment'),
 ('body',f"39 mathematical/interface contracts pass; maximum finite-difference error {r['maximum_finite_difference_error']:.3g}. The existing integrated fixed-predicate witness reaches 100% on 256 synthetic DEV rows from 64 FIT rows with 6,370 parameters. Predicate discovery and threshold learning are proposed extensions. Tree-like conditional expressivity does not prove cheaper candidate discovery or superiority on every typed target."),
 ('body','Next: one integrated learned-predicate intervention, typed residual versus bit-only comparisons, real mixed-type tables against trees and type-aware neural controls under equal tuning/work, then shared-core transfer and retention. Existing banknote results favor trees; FAS tree references are absent. All original results remain visible.'),
 ('body','Primary context: Grinsztajn et al. (arXiv:2207.08815), FT-Transformer (2106.11959), CatBoost (1706.09516) and NODE (1909.06312). Typed feature maps and neural/tree hybrids have precedents. Full derivation and linked sources: experiments/theory/TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md.'),
]
flow=[]
for kind,value in blocks:
 flow.extend([Paragraph(html.escape(value),title if kind=='title' else sub if kind=='sub' else body),Spacer(1,7)])
appendix=folder/'theory_page.pdf'
SimpleDocTemplate(str(appendix),pagesize=A4,leftMargin=45,rightMargin=45,topMargin=35,bottomMargin=35).build(flow)
temporary=folder/'published.pdf'
with pymupdf.open(pdf) as old,pymupdf.open(appendix) as add:
 old_count=len(old);added=len(add);assert added==1
 old.insert_pdf(add);old.save(temporary,garbage=0,deflate=False)
with pymupdf.open(temporary) as new,pymupdf.open(folder/'previous.pdf') as old:
 assert len(new)==len(old)+1
 assert all(new[i].get_text()==old[i].get_text() for i in range(len(old)))
 text=new[-1].get_text();assert all(v in text for v in ('39 mathematical','100%','branch utility','nominal category'))
 for x0,y0,x1,y1,*_ in new[-1].get_text('blocks'):assert min(x0,y0)>=0 and x1<=new[-1].rect.width and y1<=new[-1].rect.height
 new[-1].get_pixmap().save(str(folder/'preview.png'))
assert peak()<300000 and available()>=8192*1024 and sha(pdf)==previous
shutil.copy2(temporary,pdf)
record=dict(status='completed',previous_pages=old_count,published_pages=old_count+1,
 previous_sha256=previous,output_sha256=sha(pdf),prior_page_text='Identical on every previous page',
 peak_rss_kib=peak(),wall_s=time.monotonic()-started,
 source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),receipt)},
 scope='Typed mathematical/interface theory and prior synthetic integration; bounded rendering, no model fit/inference or tree benchmark.')
(ROOT/'report/publication_typed_semantics_theory_20261006_v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
