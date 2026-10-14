"""Completed strongest-parent continuation; excludes pending depth scores."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 paths=[ROOT/'experiments/results/token_language/aws_protected_depth_parent_2m_20261014_s6_v1.json',ROOT/'experiments/results/diagnostics/aws_language_quality_work_frontier_20261014_v1.json'];run,frontier=[json.loads(p.read_text()) for p in paths];assert run['status']==frontier['status']=='completed';old=next(x for x in frontier['rows'] if x['tag']=='aws_multidepth_readout_8m_20261013_s6_v1');assert run['arm']=='parent' and run['scored']['targets']==8192 and run['scored']['window']==1024 and not run['scored']['eos_state_reset'];new=run['scored']['nll'];assert new<old['paired_dev_nll']
 packet=dict(status='completed',old_nll=old['paired_dev_nll'],new_nll=new,gain_nats=old['paired_dev_nll']-new,added_fit_targets=run['added_targets'],warmup_targets=run['warmup_targets'],calibration_targets=run['calibration_targets'],incremental_training_s=run['added_training_s'],inherited_training_s=old['full_training_s'],scope='Single seed, same8192 reusedDEV targets and1024history. Added acquisition plus calibratedlocal clipping jointly; not clipping-only or added-depth attribution. No publicTransformer or physicalhardware win. Pilot/calibration/scoring costs additionally recorded in source result.',input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 (ROOT/'investment/protected_language_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
if __name__=='__main__':main()
