#!/usr/bin/env python3
"""Build the Kile-editable sources; never edit the sources during a build."""
import argparse,hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',default='all');p.add_argument('--us',action='store_true');a=p.parse_args()
 families=json.loads((ROOT/'portfolio.json').read_text());selected=[x for x in families if a.family in ['all',x['id']]]
 if not selected:raise SystemExit('Unknown family')
 output=ROOT/'pdf';output.mkdir(exist_ok=True);env=os.environ.copy();local=ROOT.parents[1]/'experiments/.texmf_local'
 env.update(TEXMFHOME=str(local),TEXMFVAR=str(local/'var'),TEXMFCONFIG=str(local/'config'))
 results=[]
 with tempfile.TemporaryDirectory(prefix='patent-build-') as folder:
  work=Path(folder)
  for pth in ROOT.iterdir():
   if pth.suffix in ['.tex','.cls','.sty']:shutil.copy2(pth,work/pth.name)
  for family in selected:
   pid=family['id']
   for stem in [pid,pid+'-drawings']:
    source=work/(stem+'.tex')
    if a.us and stem==pid:source.write_text(source.read_text().replace(r'\documentclass[pct]',r'\documentclass[us]'))
    for _ in range(2):
     cp=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',source.name],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
     if cp.returncode:
      print(cp.stdout[-7000:]);raise SystemExit('LaTeX failed: '+stem)
    log=(work/(stem+'.log')).read_text(errors='replace')
    if 'Overfull \\hbox' in log or 'Overfull \\vbox' in log:raise SystemExit('Layout overflow: '+stem+'\n'+ '\n'.join(x for x in log.splitlines() if 'Overfull' in x))
    suffix='-us' if a.us and stem==pid else ''
    target=output/(stem+suffix+'.pdf');shutil.copy2(work/(stem+'.pdf'),target)
    results.append(dict(file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
 print(json.dumps(dict(status='built',files=results),indent=2))
if __name__=='__main__':main()
