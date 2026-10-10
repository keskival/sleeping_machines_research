"""B3 whole-manifest metadata differs; payload identity must remain exact."""
import json
import pytest
from experiments.fas.payload_identity import digest,verify_identity


def test_generation_timers_are_the_only_allowed_difference(tmp_path):
    names=('train_clean','val_clean','val_faulty','test_clean','test_faulty')
    local=dict(name='frozen',args=dict(K=2,seed_base=20261005),generator_sha256='source',started=1.,
               splits={n:dict(sha256=n,runs=1000,wall_s=2.) for n in names})
    reference=json.loads(json.dumps(local));reference['started']=99.
    for row in reference['splits'].values():row['wall_s']=30.
    p,q=tmp_path/'local.json',tmp_path/'reference.json'
    p.write_text(json.dumps(local));q.write_text(json.dumps(reference))
    assert digest(p)!=digest(q)
    assert verify_identity(p,q,digest(p),digest(q))
    with pytest.raises(ValueError,match='digest'):verify_identity(p,q,digest(p),'wrong original fitted hash')
    reference['splits']['test_faulty']['sha256']='different dataset'
    q.write_text(json.dumps(reference))
    with pytest.raises(ValueError,match='payload'):verify_identity(p,q,digest(p),digest(q))


def test_recipe_source_and_counts_cannot_be_ignored(tmp_path):
    names=('train_clean','val_clean','val_faulty','test_clean','test_faulty')
    original=dict(args=dict(K=2),generator_sha256='source',splits={n:dict(sha256=n,runs=1000) for n in names})
    p,q=tmp_path/'local.json',tmp_path/'reference.json';p.write_text(json.dumps(original))
    for mutation in ('args','source','count'):
        changed=json.loads(json.dumps(original))
        if mutation=='args':changed['args']['K']=3
        elif mutation=='source':changed['generator_sha256']='other generator'
        else:changed['splits']['train_clean']['runs']=999
        q.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='payload'):verify_identity(p,q,digest(p),digest(q))


def test_reference_scorer_accepts_bound_timer_only_manifest_equivalence(tmp_path):
    from experiments.fas.score_sealed_v3 import verify_admission
    names=('train_clean','val_clean','val_faulty','test_clean','test_faulty')
    directory=tmp_path/'experiments/data/fas/frozen';directory.mkdir(parents=True)
    for name in names:(directory/(name+'.npz')).write_bytes(b'opaque bytes, no array loading')
    manifest=dict(name='frozen',args=dict(K=2),started=1.,
                  splits={n:dict(sha256=digest(directory/(n+'.npz')),wall_s=2.) for n in names})
    local=directory/'manifest.json';local.write_text(json.dumps(manifest))
    reference=tmp_path/'aws_manifest.json';manifest['started']=99.;reference.write_text(json.dumps(manifest))
    source=tmp_path/'model.py';source.write_text('model source')
    cp=tmp_path/'checkpoints/selected.pt';cp.parent.mkdir();cp.write_bytes(b'opaque selected weights')
    result=dict(status='completed',args=dict(data='frozen',seed=0,tag='selected',model='transformer',d=128,
                layers=2,lr=.003,no_test=True),source_sha256={'model.py':digest(source)},
                selected_weights=dict(path='checkpoints/selected.pt',sha256=digest(cp)),data_manifest_sha256=digest(reference))
    path=tmp_path/'fit.json';path.write_text(json.dumps(result))
    admission=dict(kind='transformer',data='frozen',data_manifest_sha256=digest(local),
                   source_sha256=result['source_sha256'],result='fit.json',result_sha256=digest(path),
                   seed=0,trained_tag='selected',checkpoint='checkpoints/selected.pt',checkpoint_sha256=digest(cp),
                   reference_manifest='aws_manifest.json')
    assert verify_admission(admission,root=tmp_path)[0]['data_manifest_sha256']==digest(reference)
    reference.write_text(reference.read_text()+'\n')
    with pytest.raises(ValueError,match='digest'):verify_admission(admission,root=tmp_path)
