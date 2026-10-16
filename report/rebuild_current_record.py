"""Refresh source-bound benchmark front matter without losing current appendices.

The older generated build reconstructs only its known experiment modules. This
mode renders the maintained combined Markdown record, including newer publisher
sections, rather than silently dropping them. All volumes use the same renderer.
"""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def build():
    from report.readable_report import benchmark_record_page,markdown
    from report.split_report import render_pdf
    path=ROOT/'REPORT.md';text=path.read_text();start=text.index('\n## Public benchmark record');end=text.index('\n## ',start+4)
    text=text[:start+1]+markdown([benchmark_record_page()])+'\n'+text[end:]
    path.write_text(text)
    subprocess.run([sys.executable,str(ROOT/'report/split_report.py')],cwd=ROOT,check=True)
    pages=render_pdf(path,ROOT/'report/sleeping_machines_status.pdf','Sleeping Machines — complete research record')
    print('Rendered complete current record:',pages,'pages;all maintained appendices preserved')
if __name__=='__main__':build()
