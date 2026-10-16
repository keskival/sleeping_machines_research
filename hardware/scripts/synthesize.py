#!/usr/bin/env python3
"""Generic synthesis diagnostics only; not device utilization or tapeout."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 records=[]
 with tempfile.TemporaryDirectory(prefix='sm-synth-') as folder:
  for src in sorted((ROOT/'hardware/rtl').glob('*.sv')):
   report=Path(folder)/(src.stem+'.json')
   command=f'read_verilog -sv {src}; hierarchy -check -top {src.stem}; synth -top {src.stem} -noabc; check -assert; tee -o {report} stat -json'
   cp=subprocess.run(['yosys','-Q','-p',command],capture_output=True,text=True,timeout=180)
   if cp.returncode:raise RuntimeError(cp.stdout[-3500:]+cp.stderr[-1500:])
   data=json.loads(report.read_text());m=data['modules']['\\'+src.stem]
   records.append(dict(module=src.stem,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),generic_cells=m['num_cells'],generic_cell_types=m['num_cells_by_type'],wire_bits=m['num_wire_bits']))
 packet=dict(status='passed',tool=subprocess.check_output(['yosys','-V'],text=True).strip(),records=records,scope='Yosys generic synthesis with default draft parameters,no ABC technology mapping. Counts are not FPGA LUT/BRAM/DSP utilization,ASIC area,post-route frequency,latency or power. No flagship-model numerical equivalence claim.')
 (ROOT/'hardware/evidence/generic_synthesis.json').write_text(json.dumps(packet,indent=2)+'\n');print(json.dumps(packet,indent=2))
if __name__=='__main__':main()
