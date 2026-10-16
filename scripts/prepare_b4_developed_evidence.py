"""Bind the completed developed B4 win and native hierarchy diagnosis."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    inputs={}
    def load(name):
        p=ROOT/name;inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest();r=json.loads(p.read_text());assert r['status']=='completed';return r
    wiki=load('experiments/results/tpp_b4/aws_b4x_wiki_test_20261016_v2.json');assert wiki['beats_composite_target'] and len(wiki['test_splits'])==5
    assert all(r['total']<wiki['composite_target'] for r in wiki['test_splits'])
    rows=[]
    for i in range(5):
        row=load(f'experiments/results/tpp_b4/aws_b4x_wiki_ptr_20261015_v1_s{i}.json');rows.append(row['parameters'])
    diagnosis=load('experiments/results/hierarchy/aws_g_hier_grounding_diagnosis_20261016_v1.json')
    joint=diagnosis['rows']['joint']['calibrated_root_geometry_mse'];frozen=diagnosis['rows']['decoder_only']['calibrated_root_geometry_mse']
    packet=dict(status='completed',wikipedia=dict(total=wiki['test_mean']['total'],se=wiki['test_total_se'],time=wiki['test_mean']['L_T'],marks=wiki['test_mean']['L_M'],target=wiki['composite_target'],best_single=wiki['best_single_published'],margin=wiki['composite_target']-wiki['test_mean']['total'],splits=wiki['test_splits'],selected=wiki['chosen_variant'],parameters=rows,scope='Developed attempt,one seed per five official splits;variant selected by aggregate validation only;TEST scored once. Recorded-data density scoring,not a recording-cell/dequantized comparison. Repair follows published frozen-transfer failure;no hardware/matched-compute claim.'),hierarchy=dict(rows=diagnosis['rows'],root_geometry_error_reduction=1-joint/frozen,decision=diagnosis['decision'],scope=diagnosis['scope']),input_sha256=inputs)
    for dataset,tag in [('github','aws_b4x_github_test_20261016_v3'),('stackoverflow','aws_b4x_so_test_20261016_v2')]:
        r=load(f'experiments/results/tpp_b4/{tag}.json');assert len(r['test_splits'])==5
        packet[dataset]=dict(total=r['test_mean']['total'],se=r['test_total_se'],target=r['composite_target'],best_single=r['best_single_published'],margin=r['composite_target']-r['test_mean']['total'],beats_composite=r['beats_composite_target'],splits=r['test_splits'],selected=r['chosen_variant'],scope=packet['wikipedia']['scope'])
    (ROOT/'investment/b4_developed_evidence.json').write_text(json.dumps(packet,indent=2)+'\n');print('Bound developed Wikipedia WIN and representation-versus-transfer diagnosis')
if __name__=='__main__':main()
