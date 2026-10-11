"""Source-guarded current investor artifacts; receipts stay under .git."""
import argparse,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    with open('/tmp/aws-language-publication.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
        assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
        python=str(ROOT/'.venv-docker/bin/python')
        subprocess.run([python,'scripts/prepare_forward_credit_evidence.py'],cwd=ROOT,check=True)
        subprocess.run([python,'scripts/build_pitch_deck.py','--notes'],cwd=ROOT,check=True)
        subprocess.run([python,'scripts/build_pitch_deck.py','--tag',a.tag],cwd=ROOT,check=True)
        subprocess.run([python,'scripts/build_investment_case.py','--tag',a.tag],cwd=ROOT,check=True)
if __name__=='__main__':main()
