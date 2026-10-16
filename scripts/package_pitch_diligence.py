"""Build the current private diligence bundle;local receipts,git history only."""
import argparse,hashlib,json,os,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def package(tag):
 if not tag or Path(tag).name!=tag:raise ValueError('Fresh plain tag required')
 stage=ROOT/'.git/diligence-preview'/tag;stage.mkdir(parents=True,exist_ok=False)
 paths=set()
 for pattern in ['investment/*.md','investment/*.json','investment/*.csv','investment/*.pdf','investment/EIC_ACCELERATOR/*.md','report/I_SCIENCE.*','report/II_METHODS.*','report/III_RECORD.*','report/model_family_*.md']:
  paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())
 paths.update(['README.md','REPORT.md','report/sleeping_machines_status.pdf','scripts/package_pitch_diligence.py','scripts/prepare_public_benchmark_headlines.py','scripts/verify_investor_evidence.py','scripts/public_headline_consistency.py','scripts/build_pitch_deck.py','scripts/build_investment_case.py','investment/brief/evidence_brief.tex','investment/brief/make_brief_figures.py'])
 for name in ['public_benchmark_headlines.json','b4_developed_evidence.json','bound_hierarchy_evidence.json']:
  packet=json.loads((ROOT/'investment'/name).read_text());assert packet['status']=='completed'
  for p,digest in packet.get('input_sha256',{}).items():
   file=ROOT/p
   if not file.is_file() or sha(file)!=digest:raise ValueError('Changed current evidence:'+p)
   paths.add(p)
 if any(Path(p).is_absolute() or '..' in Path(p).parts or '/archive/' in p for p in paths):raise ValueError('Unsafe/superseded member')
 hashes={p:sha(ROOT/p) for p in sorted(paths)}
 total=sum((ROOT/p).stat().st_size for p in paths)
 if total>32*1024*1024:raise ValueError('Bundle exceeds bounded32MiB input size')
 manifest=dict(classification='PRIVATE REVIEW;no external distribution',packaged_utc=datetime.now(timezone.utc).isoformat(),public_benchmark_wins=13,file_sha256=hashes,total_input_bytes=total,scope='Current materials and source-bound evidence. No model weights,credentials,contacts,document archives or publication receipt history.')
 target=ROOT/'investment/sleeping_machines_private_diligence.zip';temp=stage/target.name
 with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  z.writestr('PRIVATE_PACK_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
  for p in sorted(paths):z.write(ROOT/p,p)
 with zipfile.ZipFile(temp) as z:
  assert all(hashlib.sha256(z.read(p)).hexdigest()==digest for p,digest in hashes.items())
 assert hashes=={p:sha(ROOT/p) for p in sorted(paths)},'Concurrent bundle input change'
 os.replace(temp,target)
 (stage/'publication.json').write_text(json.dumps(dict(manifest,archive_sha256=sha(target)),indent=2)+'\n')
 print(json.dumps(dict(bundle=str(target),members=len(paths)+1,bytes=target.stat().st_size)))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);package(p.parse_args().tag)
