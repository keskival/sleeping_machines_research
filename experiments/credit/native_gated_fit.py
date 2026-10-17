"""A source-bound R1 gate followed by exact checkpoint continuation, one trainer at a time."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args()
    q=ROOT/a.candidate
    subprocess.run([sys.executable,'scripts/check_native_coverage.py',str(q)],cwd=ROOT,check=True)
    command=shlex.split([x for x in q.read_text().splitlines() if x.strip() and not x.startswith('#')][-1]);tag=command[0];assert command[1]=='experiments/r1_rmr.py'
    args=command[2:];assert '--max-steps' not in args
    names=['experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/native_slot_memory.py','experiments/credit/native_gated_fit.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    def run(extra):
        for n,d in hashes.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==d,n
        rows=[]
        with subprocess.Popen([sys.executable,command[1],*args,*extra],cwd=ROOT,stdout=subprocess.PIPE,text=True,bufsize=1) as proc:
            for line in proc.stdout:
                print(line,end='',flush=True)
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
            assert proc.wait()==0,'native phase failed; checkpoint/log retained'
        return rows
    rows=run(['--max-steps','16','--checkpoint-every','16'])
    gate=next(row for row in rows if row.get('smoke_steps')==16)
    assert gate['max_rss_kb']<2900000,'gate requires reduced-concurrency reservation'
    out=ROOT/f'experiments/results/diagnostics/{tag}_admission.json'
    out.write_text(json.dumps(dict(status='completed',battle='R1',stage='native gate before exact continuation',source_sha256=hashes,gate='PASS',measurements=gate,checkpoint=f'experiments/results/token_language/{tag}.resume.pt'),indent=2)+'\n')
    print('GATE PASS; continuing same checkpoint without repeating gate updates',flush=True)
    run(['--max-steps','0','--checkpoint-every','256'])
if __name__=='__main__':main()
