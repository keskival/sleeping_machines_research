#!/usr/bin/env python3
"""Validate compiled layout structure and source/abstract/claim inventory."""
import hashlib,json,re,zipfile
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parent

def main():
 records=[]
 for item in json.loads((ROOT/'portfolio.json').read_text()):
  pid=item['id'];tex=(ROOT/(pid+'.tex')).read_text();claims=tex.split('\\begin{patentclaims}',1)[1].split('\\end{patentclaims}',1)[0].count('\\item ')
  assert claims==item['claims'],pid
  abstract=tex.split('\\abstractstart',1)[1].split('\\end{document}',1)[0].strip();assert 50<=len(abstract.split())<=150,pid
  pages=[]
  for stem in [pid,pid+'-drawings']:
   pdf=ROOT/'pdf'/(stem+'.pdf');doc=pymupdf.open(pdf)
   assert all(abs(p.rect.width-595.276)<1 and abs(p.rect.height-841.89)<1 for p in doc),stem
   if stem==pid:
    claim_page=[i for i,p in enumerate(doc) if re.search(r'^CLAIMS$',p.get_text(),re.M)];abstract_page=[i for i,p in enumerate(doc) if re.search(r'^ABSTRACT$',p.get_text(),re.M)]
    assert len(claim_page)==len(abstract_page)==1 and 0<claim_page[0]<abstract_page[0],stem
    for page in doc:
     assert page.get_text().strip(),stem
     assert all(0<=word[0]<word[2]<=page.rect.width and 0<=word[1]<word[3]<=page.rect.height for word in page.get_text('words')),stem
   else:assert len(doc)==2 and all(('FIG. '+str(i+1)) in page.get_text() for i,page in enumerate(doc)),stem
   pages.append(len(doc))
  records.append(dict(id=pid,claims=claims,abstract_words=len(abstract.split()),text_pages=pages[0],drawing_sheets=pages[1]))
 packet=dict(status='passed',families=records,scope='Compilation/layout/inventory validation only;not office acceptance,novelty,claim support or filing clearance.')
 (ROOT/'validation.json').write_text(json.dumps(packet,indent=2)+'\n')
 # Current portable Kile package; no applicant credentials or private session logs.
 target=ROOT.parent/'sleeping_machines_patent_latex.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(ROOT.rglob('*')):
   if p.is_file() and p.suffix in ['.tex','.cls','.sty','.py','.md','.json','.pdf']:z.write(p,'patents/'+str(p.relative_to(ROOT)))
 print(json.dumps(packet,indent=2))
if __name__=='__main__':main()
