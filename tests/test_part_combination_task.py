import numpy as np
from experiments.hierarchy.part_combination_task import render,scenes,audit


def test_combinations_disjoint_children_supported_and_nuisances_balanced():
    summary=audit(scenes())
    assert summary['train']['scenes']==128
    assert summary['dev']['scenes']==summary['test']['scenes']==64


def test_root_relation_changes_label_but_not_bag_counts_or_time():
    for left,right in [(0,1),(3,7),(5,0)]:
        a,b=render(left,right,0),render(left,right,1)
        assert a.label!=b.label and a.digest!=b.digest
        assert np.array_equal(a.events[:,[0,3]],b.events[:,[0,3]])
        assert np.array_equal(np.sort(a.events[:,1]),np.sort(b.events[:,1]))
        assert np.array_equal(np.sort(a.events[:,2]),np.sort(b.events[:,2]))
        # Cross-axis histograms also cannot distinguish root swapping.
        assert np.all((a.events[:,1:3]>=0)&(a.events[:,1:3]<128))


def test_anonymous_stream_identifies_arrangement_without_label_collisions():
    rows=scenes(range(1));assert len({r.digest for r in rows})==128
    for r in rows:
        events,label=r.model_input()
        assert events.shape==(32,4) and label in (0,1)
        events[:]=0
        assert r.events.any()  # copied interface; no latent codes supplied


def test_corpus_withholds_nuisance_tapes_and_train_relations_span_three_bits():
    from experiments.hierarchy.part_combination_task import corpus
    rows=corpus();audit(rows)
    tapes=[{r.nuisance for r in rows if r.split==s} for s in ('train','dev','test')]
    assert not tapes[0]&tapes[1] and not tapes[0]&tapes[2] and not tapes[1]&tapes[2]
    # TRAIN differences contain an affine basis, avoiding an unobserved bit
    # whose effect on the target would be unconstrained by all training labels.
    differences={r.left^r.right for r in rows if r.split=='train'}
    assert differences=={0,1,3,7}
    span={0}
    for v in differences:span|={x^v for x in list(span)}
    assert span==set(range(8))
