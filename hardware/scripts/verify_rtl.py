#!/usr/bin/env python3
"""Deterministic bit-level simulation contracts; no fitted model or chip claims."""
import hashlib,json,random,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RTL=ROOT/'hardware/rtl'
HEADER='''`timescale 1ns/1ps
module tb;
reg clk=0; always #5 clk=~clk;
reg rst_n=0;
task tick;begin @(posedge clk);#1;end endtask
'''

def simulate(name,decl,body,folder):
 source=folder/(name+'_tb.sv');source.write_text(HEADER+decl+'\ninitial begin\n'+body+'\n$display("PASS '+name+'");$finish;end\ninitial begin #1000000;$fatal(1,"timeout");end\nendmodule\n')
 exe=folder/name
 cp=subprocess.run(['iverilog','-g2012','-s','tb','-o',str(exe),str(source),str(RTL/(name+'.sv'))],capture_output=True,text=True)
 if cp.returncode:raise RuntimeError(cp.stderr)
 cp=subprocess.run(['vvp',str(exe)],capture_output=True,text=True)
 if cp.returncode or 'PASS '+name not in cp.stdout:raise RuntimeError(cp.stdout+cp.stderr)
 return next(line for line in cp.stdout.splitlines() if line.startswith('PASS '))

def main():
 rng=random.Random(16001);results=[]
 with tempfile.TemporaryDirectory(prefix='sm-rtl-') as folder:
  f=Path(folder)
  decl='''reg iv=0,last=0,ordy=0; wire irdy,ov;reg [63:0] due=0,tag=0;reg [15:0] addr=0;wire [63:0] odue,otag;wire [15:0] oaddr;
sm_race_select dut(clk,rst_n,iv,irdy,last,due,addr,tag,ov,ordy,odue,oaddr,otag);'''
  body='tick;rst_n=1;tick;\n'
  for batch in range(32):
   rows=[(rng.randrange(1,20),i,rng.getrandbits(64)) for i in range(rng.randrange(1,10))];winner=min(rows,key=lambda x:x[0])
   for i,(due,addr,tag) in enumerate(rows):body+=f'iv=1;last={int(i==len(rows)-1)};due=64\'d{due};addr={addr};tag=64\'d{tag};if(!irdy)$fatal;tick;\n'
   body+=f'iv=0;if(!ov || odue!=64\'d{winner[0]} || oaddr!={winner[1]} || otag!=64\'d{winner[2]})$fatal(1,"race mismatch");\n'
   body+='repeat(3)begin if(irdy)$fatal;tick;end ordy=1;tick;ordy=0;tick;\n'
  results.append(simulate('sm_race_select',decl,body,f))
  decl='''reg cv=0,cw=0,rr=0;wire cr,rv;reg [2:0] ca=0;reg [63:0] ct=0;reg [1:0] ce=0;reg [31:0] cd=0;wire [2:0] status;wire [31:0] data;wire [63:0] tm;wire [1:0] ver;
sm_versioned_state_bank #(.ENTRIES(5),.ADDR_W(3),.DATA_W(32),.VERSION_W(2)) dut(clk,rst_n,cv,cr,cw,ca,ct,ce,cd,rv,rr,status,data,tm,ver);'''
  body='tick;rst_n=1;tick;\n'
  cases=[(0,0,0,0,0,1,0,0,0),(1,0,10,0,42,0,42,10,1),(0,0,9,0,0,3,0,0,0),(1,0,11,0,99,2,0,0,0),(0,0,12,0,0,0,42,10,1),(1,0,9,1,66,3,0,0,0),(1,0,10,1,43,0,43,10,2),(1,0,11,2,44,0,44,11,3),(1,0,12,3,45,4,0,0,0),(0,7,20,0,0,5,0,0,0),(1,4,50,0,999,0,999,50,1),(0,0,80,0,0,0,44,11,3)]
  for wr,addr,t,expected,val,status,data,tm,ver in cases:
   body+=f'cw={wr};ca={addr};ct={t};ce={expected};cd={val};cv=1;if(!cr)$fatal;tick;cv=0;if(!rv || status!={status} || data!={data} || tm!={tm} || ver!={ver})$fatal(1,"bank mismatch");repeat(2)tick;rr=1;tick;rr=0;tick;\n'
  results.append(simulate('sm_versioned_state_bank',decl,body,f))
  decl='''reg iv=0,ordy=0;wire irdy,ov,sat;reg [63:0] vals=0;reg [31:0] cs=0,ss=0;wire [63:0] outvals;
sm_temporal_transport #(.PAIRS(2),.W(16),.FRAC(8)) dut(clk,rst_n,iv,irdy,vals,cs,ss,ov,ordy,outvals,sat);'''
  body='tick;rst_n=1;tick;\n'
  for case in range(64):
   xs=[rng.randrange(-32768,32768) for _ in range(4)];c=[rng.randrange(-512,513) for _ in range(2)];s=[rng.randrange(-512,513) for _ in range(2)]
   raw=[]
   for i in range(2):raw.extend([(xs[2*i]*c[i]-xs[2*i+1]*s[i])>>8,(xs[2*i]*s[i]+xs[2*i+1]*c[i])>>8])
   ys=[min(32767,max(-32768,y)) for y in raw];overflow=int(ys!=raw)
   pack=lambda seq:sum((x&65535)<<(16*i) for i,x in enumerate(seq))
   body+=f'vals=64\'h{pack(xs):016x};cs=32\'h{pack(c):08x};ss=32\'h{pack(s):08x};iv=1;tick;iv=0;while(!ov)tick;if(outvals!==64\'h{pack(ys):016x} || sat!={overflow})$fatal(1,"transport mismatch case{case}");repeat(2)tick;ordy=1;tick;ordy=0;tick;\n'
  results.append(simulate('sm_temporal_transport',decl,body,f))
  decl='''reg pv=0,ar=0,gv=0,rr=0;wire pr,av,gr,rv;reg [15:0] pp=0,pc=0;reg [31:0] ppv=0,pcv=0,pose=0;reg [63:0] ca=0,issued=0,pa=0,cut=0;reg [1:0] slot=0;wire [1:0] aslot;wire [2:0] astat,rstat;wire [15:0] rp,rc;wire [31:0] rpv,rcv,rpose;wire [63:0] ri;
sm_binding_directory #(.ENTRIES(3),.SLOT_W(2)) dut(clk,rst_n,pv,pr,pp,pc,ppv,pcv,ca,issued,pa,pose,av,ar,astat,aslot,gv,gr,slot,cut,rv,rr,rstat,rp,rc,rpv,rcv,ri,rpose);'''
  body='tick;rst_n=1;tick;\n'
  # Same pose, distinct children; duplicate idempotence, mutation rejection,
  # parent-version distinction, full capacity, unavailable reference.
  puts=[(1,10,1,2,1,2,3,55,0,0),(1,11,1,4,1,2,3,55,0,1),(1,10,1,2,1,2,3,55,0,0),(1,10,1,3,1,2,3,55,2,0),(1,10,2,3,1,2,4,55,0,2),(2,12,1,1,1,2,3,55,4,0),(2,12,1,1,5,2,3,55,3,0)]
  for parent,child,pver,cver,available,it,pt,pose,status,slot in puts:
   body+=f'pp={parent};pc={child};ppv={pver};pcv={cver};ca={available};issued={it};pa={pt};pose={pose};pv=1;if(!pr)$fatal;tick;pv=0;if(!av || astat!={status} || aslot!={slot})$fatal(1,"binding put");ar=1;tick;ar=0;tick;\n'
  for slot,cut,status,child,pver,cver in [(0,2,3,0,0,0),(0,3,0,10,1,2),(1,3,0,11,1,4),(2,4,0,10,2,3),(3,9,1,0,0,0)]:
   body+=f'slot={slot};cut={cut};gv=1;if(!gr)$fatal;tick;gv=0;if(!rv || rstat!={status} || rc!={child} || rpv!={pver} || rcv!={cver})$fatal(1,"binding get");repeat(2)tick;rr=1;tick;rr=0;tick;\n'
  results.append(simulate('sm_binding_directory',decl,body,f))
  hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RTL.glob('*.sv'))}
  hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  packet=dict(status='passed',tool=subprocess.check_output(['iverilog','-V'],stderr=subprocess.DEVNULL,text=True).splitlines()[0],tests=results,coverage=dict(race_batches=32,state_commands=len(cases),transport_vectors=64,binding_puts=len(puts),binding_reads=5),source_sha256=hashes,scope='RTL simulation only,bit-level declared integer interface. No flagship prediction parity,FPGA utilization,physical latency or energy measurement.')
  out=ROOT/'hardware/evidence/rtl_contracts.json';out.write_text(json.dumps(packet,indent=2)+'\n');print(json.dumps(packet,indent=2))
if __name__=='__main__':main()
