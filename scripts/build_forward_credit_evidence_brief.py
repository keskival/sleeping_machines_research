"""Rebuild the current evidence brief from its LaTeX source; private local artifact."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
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
        build=preview/'build';shutil.copytree(ROOT/'investment/brief',build)
        env=os.environ.copy();tex=ROOT/'experiments/.texmf_local'
        if tex.is_dir():
            env.update(TEXMFHOME=str(tex),TEXMFVAR=str(tex/'var'),TEXMFCONFIG=str(tex/'config'))
        with (preview/'build.log').open('w') as log:
            subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','evidence_brief.tex'],cwd=build,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        pdf=build/'evidence_brief.pdf'
        text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
        for expected in ['98.4','2.09','77.8','11 October 2026','working research prototype']:
            assert expected in text,expected
        (preview/'text.txt').write_text(text)
        output=ROOT/'investment/sleeping_machines_evidence_brief.pdf';shutil.copyfile(pdf,output)
        info=subprocess.check_output(['pdfinfo',str(output)],text=True)
        (preview/'pdfinfo.txt').write_text(info)
        subprocess.run(['pdftoppm','-f','10','-singlefile','-scale-to','1400','-png',str(output),str(preview/'forward-credit')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        record=dict(status='completed',wall_s=time.monotonic()-started,artifact=str(output.relative_to(ROOT)),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((ROOT/'investment/brief/evidence_brief.tex').read_bytes()).hexdigest(),packet_sha256=hashlib.sha256((ROOT/'investment/forward_credit_evidence_20261011.json').read_bytes()).hexdigest(),scope='Current private evidence brief; source-bound component evidence and conditional potential; no distribution.')
        (preview/'publication.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
