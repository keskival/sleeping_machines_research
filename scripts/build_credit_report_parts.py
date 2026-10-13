"""Publish current source-bound credit chapter and the three canonical report parts.
The combined graphical builder needs AWS-local parent JSONs; when absent,
retain its last valid PDF and update only the completed credit chapter in the
published master, using the same chapter/Markdown functions as that builder.
"""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'report'))
from report.optimizer_transport_appendix import pages,TITLE
from report.readable_report import markdown


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    receipt=ROOT/'.git/credit-report-preview'/a.tag/'publication.json'
    assert not receipt.exists();receipt.parent.mkdir(parents=True,exist_ok=True)
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
        chapter=markdown(pages());assert TITLE in chapter
        master=ROOT/'REPORT.md';text=master.read_text();marker='\n## '+TITLE
        if marker in text:
            start=text.index(marker);end=text.find('\n## ',start+len(marker))
            text=text[:start]+'\n'+chapter.strip()+'\n'+('' if end<0 else text[end:])
        else:
            boundary='\n## Appendix. B1 exact deep online credit'
            assert boundary in text;text=text.replace(boundary,'\n'+chapter.strip()+'\n'+boundary,1)
        master.write_text(text)
        missing=[f'experiments/results/tpp_b4/aws_observation_anchored_readout_20261011_s{s}.json' for s in range(3) if not (ROOT/f'experiments/results/tpp_b4/aws_observation_anchored_readout_20261011_s{s}.json').exists()]
        # New AWS transfer packets also bind checkpoint binaries used only for
        # provenance verification by the graphical builder. Preserve its latest
        # valid AWS rendering when those binaries are not replicated here;
        # authored credit chapter and all canonical parts still regenerate.
        transfer=ROOT/'experiments/results/diagnostics/aws_fresh_scaling_transfer_20261013_v1.json'
        if transfer.exists():
            packet=json.loads(transfer.read_text())
            missing.extend(name for name in packet.get('input_sha256',{})
                           if name.endswith('.pt') and not (ROOT/name).exists())
        python=str(ROOT/'.venv-docker/bin/python')
        if not missing:subprocess.run([python,'report/make_pdf.py'],cwd=ROOT,check=True)
        with (receipt.parent/'split.log').open('w') as log:
            subprocess.run([python,'report/split_report.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        paths=['REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf']
        record=dict(status='completed',scope='Current source-bound credit chapter, master and canonical report parts; combined graphical PDF unchanged where AWS parent JSONs unavailable.',missing_combined_pdf_parents=missing,artifact_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
        receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
