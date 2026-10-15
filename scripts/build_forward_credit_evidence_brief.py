"""Rebuild the current evidence brief from its LaTeX source; private local artifact."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
import pymupdf as fitz
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Invalid tag')
    preview=ROOT/'.git/evidence-brief-preview'/args.tag
    assert not preview.exists();preview.mkdir(parents=True)
    started=time.monotonic()
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/prepare_forward_credit_evidence.py'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/prepare_language_learning_evidence.py'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/prepare_market_transformer_evidence.py'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/prepare_protected_language_evidence.py'],cwd=ROOT,check=True)
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/prepare_online_market_evidence.py'],cwd=ROOT,check=True)
        build=preview/'build';shutil.copytree(ROOT/'investment/brief',build)
        env=os.environ.copy();tex=ROOT/'experiments/.texmf_local'
        if tex.is_dir():
            env.update(TEXMFHOME=str(tex),TEXMFVAR=str(tex/'var'),TEXMFCONFIG=str(tex/'config'))
        with (preview/'build.log').open('w') as log:
            subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','evidence_brief.tex'],cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        pdf=build/'evidence_brief.pdf'
        document=fitz.open(pdf)
        text='\n'.join(page.get_text() for page in document)
        for expected in ['98.4','2.09','77.8','15 October 2026','working research prototype','0.09279','15.27','5.17588','5.28043','8.07854','8.10271','8.17919','0.02417','0.10066','14.7']:
            assert expected in text,expected
        (preview/'text.txt').write_text(text)
        output=ROOT/'investment/sleeping_machines_evidence_brief.pdf';shutil.copyfile(pdf,output)
        info=json.dumps(dict(pages=len(document),metadata=document.metadata),indent=2)
        (preview/'pdfinfo.txt').write_text(info)
        for label,index in [('forward-credit',9),('language-learning',next(i for i,page in enumerate(document) if '0.09279' in page.get_text() and '15.27' in page.get_text()))]:
            page=document[index];page.get_pixmap(matrix=fitz.Matrix(1400/page.rect.width,1400/page.rect.width)).save(preview/(label+'.png'))
        record=dict(status='completed',wall_s=time.monotonic()-started,artifact=str(output.relative_to(ROOT)),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/'investment/brief/evidence_brief.tex').read_bytes()).hexdigest(),packet_sha256=hashlib.sha256((ROOT/'investment/forward_credit_evidence_20261011.json').read_bytes()).hexdigest(),language_packet_sha256=hashlib.sha256((ROOT/'investment/language_learning_evidence.json').read_bytes()).hexdigest(),market_packet_sha256=hashlib.sha256((ROOT/'investment/market_transformer_evidence.json').read_bytes()).hexdigest(),scope='Current private evidence brief; source-bound component evidence and conditional potential; no distribution.')
        (preview/'publication.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
