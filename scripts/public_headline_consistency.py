"""Check the actual public-win headline in sources AND generated PDFs."""
import re
from pathlib import Path
SOURCES=['README.md','report/I_SCIENCE.md','investment/PITCH.md','investment/INVESTMENT_CASE.md','investment/INVESTOR_PROOF_PLAN.md','investment/README.md','investment/PITCH_DECK.json','investment/PITCH_DECK_NOTES.md','investment/brief/evidence_brief.tex','investment/EIC_ACCELERATOR/SHORT_PROPOSAL_DRAFT.md','investment/EIC_ACCELERATOR/EIC_ACCELERATOR_PLAN.md','investment/EIC_ACCELERATOR/PITCH_DECK_10_SLIDES.md','investment/EIC_ACCELERATOR/VIDEO_SCRIPT_3MIN.md']
PDFS=['investment/sleeping_machines_evidence_brief.pdf','investment/sleeping_machines_investment_case.pdf','investment/sleeping_machines_investor_pitch.pdf','investment/sleeping_machines_pitch_deck.pdf','investment/sleeping_machines_pitch_deck_main.pdf','report/I_SCIENCE.pdf','report/sleeping_machines_status.pdf']
def text_problem(text,wins):
 normalized=' '.join(text.replace('**','').split())
 expected=rf'\b{wins}\s+public\s+(?:benchmark|leaderboard)\s+wins\b'
 if not re.search(expected,normalized,re.I):return 'current public-win headline missing'
 if re.search(r'\b(?:nine|9)\s+public\s+(?:benchmark|leaderboard)\s+wins\b',normalized,re.I):return 'superseded nine-win headline present'
 return None

def check(root,packet):
 wins=packet['wins'];rows=[]
 assert wins==len(packet['rows']) and packet['runs']==sum(r['runs'] for r in packet['rows'])
 assert packet['ahead']==sum(r['ahead'] for r in packet['rows'])
 for name in SOURCES:
  p=root/name;error='missing document' if not p.exists() else text_problem(p.read_text(),wins)
  rows.append(dict(file=name,status='ok' if error is None else 'fail',error=error))
 import pymupdf
 for name in PDFS:
  p=root/name
  if not p.exists():error='missing PDF'
  else:
   with pymupdf.open(p) as doc:
    # The report's complete appendices contain old experiment-specific records;
    # the current global headline belongs to its generated front matter.
    pages=doc if '/report/' not in str(p) else [doc[i] for i in range(min(3,len(doc)))]
    text=' '.join(page.get_text() for page in pages);error=text_problem(text,wins)
  rows.append(dict(file=name,status='ok' if error is None else 'fail',error=error))
 return rows
