"""R1 actual-driver resource gate for the admitted 16M native depth ladder."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCES=['experiments/hierarchy/depth_data_resource_gate.py','experiments/r1_deep_keyed_stream.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--layers',type=int,required=True);a=p.parse_args();start=time.monotonic()
 command=[sys.executable,'experiments/r1_deep_keyed_stream.py','--tag',a.tag+'_actor','--layers',str(a.layers),'--train-tokens','16777216','--seed','0','--compile-blocks','1','--max-steps','32']
 child=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 print(child.stdout,flush=True);assert child.returncode==0,'actual native resource smoke failed'
 rows=[]
 for line in child.stdout.splitlines():
  try:rows.append(json.loads(line))
  except ValueError:pass
 smoke=next(x for x in reversed(rows) if x.get('smoke_steps')==32)
 passed=smoke['max_rss_kb']<2550000 and smoke['s_per_step']<15
 result=dict(status='completed' if passed else 'failed',battle='R1',stage='depth_data_resource_gate',gate='PASS' if passed else 'STOP',layers=a.layers,measurement=smoke,command=command,process_wall_s=time.monotonic()-start,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES},decision='Admit the already ordered one-pass 16M depth fit only with measured RSS margin and bounded update time.',scope='32 actual native updates,including compiler startup;no quality selection or public TEST. Resource smoke presentations charged separately from the fit;one CPU thread,pinned gym slot.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 assert passed,'resource gate requires implementation repair'
if __name__=='__main__':main()
