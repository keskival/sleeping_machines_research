"""Regenerate a conflicted derived figure from source-bound current results."""
import fcntl,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from report.deep_credit_blocks_appendix import figure,load_bound,PREFIX
def main():
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        path=ROOT/PREFIX;figure(load_bound(path),path)
        paths=['report/figures/aws_deep_credit_blocks.svg','report/figures/aws_deep_credit_blocks.png','report/figures/aws_deep_credit_blocks.inputs.json']
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','Regenerate current exact-credit figure after autosync conflict','--',*paths],cwd=ROOT,check=True)
if __name__=='__main__':main()
