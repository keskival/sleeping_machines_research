"""Private investor packet: completed engineering measurements, not quality wins."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=['experiments/results/credit/aws_same_cut_local_clip_calibration_20261014_v1.json','experiments/results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json','experiments/results/diagnostics/aws_calibrated_clip_pair_verdict_20261014_v1.json']
 records=[]
 for name in names:
  r=json.loads((ROOT/name).read_text());assert r['status']=='completed'
  for source,digest in {**r['source_sha256'],**r.get('input_sha256',{})}.items():assert sha(ROOT/source)==digest,source
  records.append(r)
 clip,readout,paired=records;metrics=clip['heldout_metrics'];a,b=readout['rows']
 packet=dict(local_clipping_fit=dict(rows=paired['rows'],nll_gain=paired['nll_gain'],incremental_work_ratio=paired['incremental_work_ratio'],timing_boundary='single-thread CPU process/scoring proxy including offline calibration; distributed synchronization costs not measured',protocol=paired['protocol'],decision=paired['decision']),status='completed',audience='private investor diligence under existing disclosure policy',evidence_level='one saved learner cut/heldout TRAIN clipping geometry; single-seed readout development',readout=dict(basic_nll=a['scored']['nll'],richer_nll=b['scored']['nll'],gain=a['scored']['nll']-b['scored']['nll']),locality=dict(heldout_metrics=metrics,adam_displacement_increase_pct=100*(metrics['calibrated1p25']['rms_adam_displacement']/metrics['global1']['rms_adam_displacement']-1),aggregate_cap_budget=1.25,placement_activation=clip['placement_activation'],audit_wall_s=clip['wall_s']),completed_evidence_sha256={n:sha(ROOT/n) for n in names},source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},scope='Readout: same8192DEVtargets/1024history/8Mseed6, additionalcapacity/work. Clipping:24TRAINcalibration/8heldout chunks on frozen completedD8actor and moments; same raw credit, calculated one-step Adam displacement. Fixed per-bank caps avoid runtime global norm. Completed paired2Mcontinuation from identical8MbankedD8actor,seed6: local6.226238/global6.231193 NLL on8192DEVtargets/1024history; calibration16kTRAINtargets and process/scoring costs charged. Fixed caps remove clipping globalnorm during training; offline calibration usesglobalinformation. Fulltrainer hasother coordination; no async hardware timing/energy or public language benchmark win.')
 (ROOT/'investment/locality_scaling_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
if __name__=='__main__':main()
