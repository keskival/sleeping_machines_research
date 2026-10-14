"""Investor evidence for completed within-family sparse-teaching quality/work."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    path=ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_gate.json'
    gate=json.loads(path.read_text());assert gate['status']=='completed' and gate['gate']=='PASS'
    parents={str(path.relative_to(ROOT)):sha(path)}
    for name,digest in {**gate['source_sha256'],**gate['input_sha256']}.items():assert sha(ROOT/name)==digest,name
    tags=['aws_integrated_fresh_scaling_8m_pool4_20261012_s6_v1','aws_bernoulli_teacher_8m_20261013_s6_v2'];rows=[]
    for tag in tags:
        p=ROOT/f'experiments/results/token_language/{tag}.json';r=json.loads(p.read_text());assert r['status']=='completed';parents[str(p.relative_to(ROOT))]=sha(p)
        rows.append(dict(tag=tag,acquired_targets=r['presentations_total'],final_dev_nll=r['curve'][-1]['dev_nll'],complete_training_s=r['train_wall_s_total']))
    assert rows[0]['acquired_targets']==rows[1]['acquired_targets']
    gain=rows[0]['final_dev_nll']-rows[1]['final_dev_nll'];reduction=100*(1-rows[1]['complete_training_s']/rows[0]['complete_training_s'])
    assert abs(gain+gate['final_nll_delta'])<1e-12
    packet=dict(status='completed',evidence_level='single-seed development at8M; three-seed confirmation at128k/512k',comparison='Within-family quality/time win over earlier full-teaching implementation',rows=rows,metrics=dict(language_learning_nll_gain=gain,language_learning_time_reduction_pct=reduction,language_learning_new_nll=rows[1]['final_dev_nll'],language_learning_old_nll=rows[0]['final_dev_nll']),teacher_work=gate['teacher_work'],completed_evidence_sha256=parents,source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__).resolve())},scope='Proper GPT2 FineWeb, same8,388,352 acquired targets/seed6/prior/update/evaluation budget and DEV scoring. New fit combines ordered token-gradient execution, exact cached suffix replay and0.25 importance-weighted actual counterfactual teaching. At8M time savings combine execution and teacher changes. Factual windowBP, all-key discovery and global clipping remain; Adam updates remain fully charged. No Transformer comparison at this scale, public-validation result, ASIC energy or asynchronous hardware timing claim.',hardware_status='Measured sparse teacher work is progress toward event-driven learning. Receiver-local versioned AdamW is a numerical/implementation integration; it does not remove full-gradient generation or global clipping. Clockless execution is compatible with local arithmetic and dependency handshakes; economical full training is a separate engineering/measurement target.')
    (ROOT/'investment/language_learning_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps(packet['metrics']))
if __name__=='__main__':main()
