"""Stop only legacy admissions; preserve trainers until lock-safe six-slot handoff."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def runners():
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if b'experiments/queue/run_safe.sh' in args:found.append(int(p.name))
 return found

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--legacy-pid',type=int,required=True);a=ap.parse_args()
 assert os.uname().nodename=='ip-172-31-47-132';os.chdir(ROOT)
 plan=Path('experiments/queue/aws_model_improvement_repair_20261005T161000Z/manifest.json')
 status=plan.parent/'worker_recovery.aws6.status.json';assert not status.exists()
 cmd=Path(f'/proc/{a.legacy_pid}/cmdline').read_bytes().split(b'\0');assert b'scripts/run_aws_product_priority.py' in cmd
 os.kill(a.legacy_pid,signal.SIGSTOP)
 print('Legacy admissions stopped; existing run_safe trainers continue',flush=True)
 while runners():time.sleep(20)
 # All training wrappers ended; release the legacy host reservation.
 os.kill(a.legacy_pid,signal.SIGTERM);os.kill(a.legacy_pid,signal.SIGCONT)
 print('Starting six slots pinned to CPU0..5; CPU6/7 and SMT siblings reserved',flush=True)
 subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/run_aws_product_priority_six.py','--manifest',str(plan)],check=True)
if __name__=='__main__':main()
