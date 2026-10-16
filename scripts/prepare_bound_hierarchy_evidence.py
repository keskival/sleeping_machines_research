"""Source-bind native versioned parts and completed fresh-data depth evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 inputs={}
 def load(name):
  p=ROOT/name;r=json.loads(p.read_text());assert r['status']=='completed'
  inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest();return r
 pre=load('experiments/results/hierarchy/aws_g_hier_bound_native_preflight_20261016_v1.json')
 contracts=load('experiments/results/hierarchy/bound_memory_contracts_20261016_v1.json')
 assert pre['gate']=='PASS' and pre['bitwise_forward_schedule_and_gradient_parity'] and pre['local_gradients_finite']
 language=[load(f'experiments/results/token_language/curie_r1_deep_keyed_stream_L{d}_4M_s0_20261015_v1.json') for d in [2,8]]
 for k in ['seed','train_tokens','lane','d','modes','heads','dk','dropout','lr','wd','batch']:
  assert language[0]['args'][k]==language[1]['args'][k],k
 assert language[0]['steps']==language[1]['steps']
 packet=dict(status='completed',native_bindings=dict(levels=pre['levels'],native_stages=pre['native_stages'],actor_parameters=pre['parameters'],actor_parameters_added=0,
  bitwise_forward_schedule_and_gradient_parity=True,contract_tests=contracts['tests'],bound_part_pool=pre['bound_part_pool'],local_race_gradients_finite=True,
  retained_records=pre['final_work']['binding_records'],removed_records=pre['garbage_collected_records'],retained_edges=pre['final_work']['binding_edges'],logical_tensor_bytes=pre['final_work']['binding_logical_tensor_bytes'],
  auxiliary_head_module_parameters=pre['root_race_head_parameters'],
  scope='One complete32event TRAIN scene,acquired native role-memory actor,eight abstraction levels/four stages each. Versioned directories preserve issued parts after lower writes;existing actor unchanged. Local race pools two bound children;arbitrary future mark tests gradient connectivity only. No semantic transfer or learned query/identity result. Metadata,extra lifetimes,read transport and auxiliary training remain paid work;logical tensor bytes are aliases,not net extra allocation.'),
 fresh_data_depth=dict(train_tokens=language[0]['args']['train_tokens'],steps=language[0]['steps'],rows=[dict(layers=r['args']['layers'],parameters=r['parameters'],nll=r['flagship_protocol']['nll'],last_quarter_prequential=r['prequential_last_quarter'],presentations=r['presentations']) for r in language],nll_gain=language[0]['flagship_protocol']['nll']-language[1]['flagship_protocol']['nll'],scope='Single seed,one pass over the same4M fresh GPT2/FineWeb tokens,equal update count and recipe,different parameters/work. Flagship8192target development protocol,not the65528target slice or sealed public reference. Actual word-initial use is supported separately by the completed role diagnostic;no semantic hierarchy or compute-frontier claim.'),input_sha256=inputs)
 audit=load('experiments/results/hierarchy/aws_g_hier_ownership_support_20261016_v1.json')
 owned=load('experiments/results/hierarchy/aws_g_hier_owned_native_preflight_20261016_v1.json')
 tests=load('experiments/results/hierarchy/owned_memory_contracts_20261016_v1.json')
 assert owned['gate']=='PASS' and owned['train_eval_bitwise_equal'] and owned['all_level_gradients_finite_and_nonzero']
 packet['ownership_engineering']=dict(fragmented_fraction_train=audit['rows']['train']['never_pure_complete_fraction'],fragmented_fraction_dev=audit['rows']['dev']['never_pure_complete_fraction'],train_scenes=audit['rows']['train']['scenes'],dev_scenes=audit['rows']['dev']['scenes'],threshold=audit['threshold'],complete_part_memories=owned['complete_pure_part_memories'],pixels_per_part=owned['pixels_per_part'],levels=owned['levels'],native_stages=owned['native_stages'],actor_parameters=owned['parameters'],contract_tests=tests['tests'],work=owned['work'],scope=owned['scope'])
 (ROOT/'investment/bound_hierarchy_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
 print('Bound native part-memory integration and one-pass depth quality evidence.')
if __name__=='__main__':main()
