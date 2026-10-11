"""Build and validate current forward-credit tutorial, receipts under local git."""
import argparse,fcntl,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic()
    scratch=ROOT/'.git/tutorial-build-preview'/a.tag;scratch.mkdir(parents=True,exist_ok=False)
    source=ROOT/'tutorials/09_forward_teaches_credit.tex'
    for p in [source,ROOT/'tutorials/tutorial.sty']:shutil.copy(p,scratch)
    with (scratch/'build.log').open('w') as log:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',source.name],cwd=scratch,stdout=log,stderr=subprocess.STDOUT,check=True)
    pdf=scratch/source.with_suffix('.pdf').name;txt=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
    assert all(s in txt for s in ['Exercises','Solution.','Where this appears'])
    pages=int(re.search(r'Pages:\s+(\d+)',subprocess.check_output(['pdfinfo',str(pdf)],text=True)).group(1));assert pages==3,pages
    log=(scratch/source.with_suffix('.log').name).read_text();assert 'Overfull' not in log
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
        target=ROOT/'tutorials'/pdf.name;shutil.copy(pdf,target)
        readme=ROOT/'tutorials/README.md';text=readme.read_text().replace('All eight planned tutorials are complete. Each is three pages.','All eight planned tutorials and the forward-credit extension are complete. Each is three pages.')
        entry='| 9 | [Forward computation teaches credit](09_forward_teaches_credit.pdf) · [source](09_forward_teaches_credit.tex) | done (3 pp.) |'
        if '| 9 |' not in text:
            lines=text.splitlines();position=next(i for i,l in enumerate(lines) if l.startswith('| 8 |'));lines.insert(position+1,entry);text='\n'.join(lines)+'\n'
        readme.write_text(text);paths=['tutorials/'+pdf.name,'tutorials/README.md']
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish forward-teaches-credit tutorial with corrected causal and cost accounting','--',*paths],cwd=ROOT,check=True)
    result=dict(status='completed',tag=a.tag,battle='B1/R1',pages=pages,wall_s=time.monotonic()-start,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    (scratch/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
