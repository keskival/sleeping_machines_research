"""Build missing tutorials in local git scratch, validate PDFs, publish current files."""
import argparse,fcntl,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();scratch=ROOT/'.git/tutorial-build-preview'/a.tag;scratch.mkdir(parents=True,exist_ok=False)
    sources=sorted((ROOT/'tutorials').glob('0[4-8]_*.tex'));assert len(sources)==5
    shutil.copy(ROOT/'tutorials/tutorial.sty',scratch);artifacts=[]
    for source in sources:
        shutil.copy(source,scratch)
        with (scratch/(source.stem+'.build.log')).open('w') as log:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',source.name],cwd=scratch,stdout=log,stderr=subprocess.STDOUT,check=True)
        pdf=scratch/(source.stem+'.pdf');txt=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
        assert 'Exercises' in txt and 'Solution.' in txt and 'Where this appears' in txt,source.name
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        texlog=(scratch/(source.stem+'.log')).read_text();overfull=[l for l in texlog.splitlines() if 'Overfull' in l];assert not overfull,(source.name,overfull)
        artifacts.append(dict(source=source.name,pdf=pdf.name,pages=pages,sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()))
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        for artifact in artifacts:shutil.copy(scratch/artifact['pdf'],ROOT/'tutorials'/artifact['pdf'])
        readme=ROOT/'tutorials/README.md';text=readme.read_text()
        names={4:'Information theory for learning signals',5:'Learning dynamics: contraction, Bellman equations and TD',6:'Counting-process martingales, Fisher information and natural gradients',7:'Spectra, scaling and queues',8:'Formal verification with Lean 4'}
        for artifact in artifacts:
            number=int(artifact['source'][:2]);pattern=rf'^\| {number} \|.*$';replacement=f'| {number} | [{names[number]}]({artifact["pdf"]}) · [source]({artifact["source"]}) | done ({artifact["pages"]} pp.) |';text=re.sub(pattern,replacement,text,flags=re.M)
        text=text.replace('## Outlines for the remaining tutorials (same format as 1–3: ~3 pages, exercises with solutions)','## Coverage and reading path')
        text=text.replace('Build: `latexmk -pdf <file>.tex` (shared style `tutorial.sty`).','Build one tutorial from `tutorials/`: `latexmk -pdf <file>.tex` (shared style `tutorial.sty`). Rebuild tutorials 4–8 with `scripts/build_tutorials.py --tag <unique-tag>`; it checks text extraction, required sections and layout before publishing. The Lean PDF is instructional; a fresh proof build requires the pinned toolchain separately.')
        readme.write_text(text)
        paths=['tutorials/'+x['pdf'] for x in artifacts]+['tutorials/README.md'];subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Publish all five missing formal-analysis tutorial PDFs and completed index','--',*paths],cwd=ROOT,check=True)
    receipt=dict(status='completed',tag=a.tag,wall_s=time.monotonic()-start,artifacts=artifacts,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[ROOT/'tutorials/tutorial.sty']});(scratch/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(artifacts,indent=2))
if __name__=='__main__':main()
