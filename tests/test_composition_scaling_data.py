import numpy as np
from experiments.hierarchy.composition_scaling_data import training_corpus,fresh_dev_corpus,broad_calibration,audit_corpus,metric_scene
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from sleeping_machines.part_identity_composition import fit_shape_vocabulary
from sleeping_machines.observed_ownership_composition import observed_groups


def test_larger_training_support_preserves_the_original_structural_holdout():
 rows=training_corpus();audit=audit_corpus(rows);assert audit['examples']==2048 and audit['abstract_arrangements']==64
 old=corpus();forbidden={(r.left,r.right,r.root_swap) for r in old if r.split!='train'}
 assert not forbidden & {(r.left,r.right,r.root_swap) for r in rows}
 assert not {r.digest for r in old}&{r.digest for r in rows}
 cal=broad_calibration(rows);assert len(cal)==64 and len({(r.left,r.right,r.root_swap) for r in cal})==64
 fresh=fresh_dev_corpus();assert len(fresh)==64 and sum(r.label for r in fresh)==32
 assert not {r.digest for r in fresh}&{r.digest for r in rows+old}


def test_geometry_variation_preserves_observed_shapes_complete_groups_and_pair_labels():
 rows=training_corpus();vocab=fit_shape_vocabulary([r.model_input()[0] for r in corpus() if r.split=='train'],10.723805294763608)
 enlarged=fit_shape_vocabulary([r.events for r in rows],10.723805294763608)
 assert enlarged==vocab
 for row in rows:
  groups,_=observed_groups(row.events[:,1:3],10.723805294763608)
  assert len(groups)==8 and all(len(g)==4 for g in groups)
 for i,j in training_pairs([r.model_input() for r in rows]):
  assert np.array_equal(rows[i].events[:,0],rows[j].events[:,0])
 # Variation changes relative spacing,not just common translation.
 extents={tuple(np.ptp(metric_scene(0,0,0,n).events[:,1:3],axis=0)) for n in range(1000,1032)}
 assert len(extents)>8
