"""B3 synthetic statistics and interruption contracts; no FAS data access."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from experiments.fas.sealed_registry import reserve, complete
from experiments.fas.stage5_statistics_v2 import analyze, auroc, average_ranks
from experiments.fas.score_sealed_v2 import verify_admission
from experiments.fas.sealed_registry import sha


def test_reservation_survives_interruption_and_output_renaming(tmp_path):
    reserve(tmp_path, 'trained_seed6', 'checkpoint_config_identity', {'checkpoint_sha256': 'abc'})
    with pytest.raises(ValueError):
        reserve(tmp_path, 'renamed_output', 'checkpoint_config_identity', {})
    with pytest.raises(FileExistsError):
        reserve(tmp_path, 'trained_seed6', 'other_identity', {})


def test_completed_ledger_and_legacy_tag_block_rescoring(tmp_path):
    reservation = reserve(tmp_path, 'trained_seed6', 'identity', {})
    result, scores = tmp_path/'result.json', tmp_path/'scores.npz'
    result.write_text('{}'); scores.write_bytes(b'synthetic')
    complete(reservation, result, scores)
    assert json.loads((tmp_path/'fas_v2_test_ledger.jsonl').read_text())['status'] == 'completed'
    with pytest.raises(ValueError): reserve(tmp_path, 'renamed_output', 'identity', {})
    with pytest.raises(ValueError): complete(reservation, result, scores)
    with (tmp_path/'fas_v2_test_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps({'tag':'legacy'})+'\n')
    with pytest.raises(ValueError): reserve(tmp_path, 'legacy', 'new_identity', {})


def test_auroc_ties_and_rank_average_are_monotone_scale_invariant():
    assert auroc([0.,1.], [1.,2.]) == .875
    pairs = [(np.array([0.,1.]), np.array([1.,2.])),
             (np.array([1.,0.]), np.array([2.,3.]))]
    changed = [(np.exp(c),np.exp(f)) for c,f in pairs]
    assert np.concatenate(average_ranks(pairs)) == pytest.approx(np.concatenate(average_ranks(changed)))


def test_reference_seed_permutation_cannot_change_bootstrap():
    good = (np.array([0.,1.,2.,3.]), np.array([4.,5.,6.,7.]))
    bad = (good[1],good[0])
    intermediate = (np.array([0.,2.,4.,6.]), np.array([1.,3.,5.,7.]))
    families = {'transformer':[good,bad,intermediate]}
    result = analyze([intermediate]*3,families,resamples=100)
    swapped = analyze([intermediate]*3,{'transformer':[bad,intermediate,good]},resamples=100)
    assert result['bootstrap_95'] == swapped['bootstrap_95']
    assert result['strongest_auroc'] == pytest.approx((1+0+.625)/3)
    # The legacy seed-0 bootstrap changes materially when the first seed changes.
    assert auroc(*good)-auroc(*bad) == 1.


def test_missing_primary_prefix_is_shared_or_refused():
    pair = (np.array([0.,1.,np.nan]),np.array([2.,3.,np.nan]))
    result = analyze([pair]*3,{'reference':[pair]},resamples=10)
    assert result['eligible_samples'] == dict(clean=2,faulty=2,excluded_short_clean=1,excluded_short_faulty=1)
    changed = (np.array([0.,np.nan,1.]),pair[1])
    with pytest.raises(ValueError): analyze([pair]*3,{'reference':[changed]},resamples=10)
    with pytest.raises(ValueError): analyze([pair]*3,{'reference':[(pair[0],np.array([2.,np.inf,np.nan]))]},resamples=10)


def test_concurrent_different_tags_cannot_reserve_the_same_model(tmp_path):
    def attempt(tag):
        try:
            reserve(tmp_path, tag, 'same_checkpoint', {})
            return 'reserved'
        except ValueError:
            return 'refused'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt,['first','second'])) == ['refused','reserved']


def test_admission_hashes_reject_data_and_source_corruption_before_loading(tmp_path):
    data = tmp_path/'experiments/data/fas/frozen'; data.mkdir(parents=True)
    splits = ('train_clean','val_clean','val_faulty','test_clean','test_faulty')
    for split in splits: (data/(split+'.npz')).write_bytes(b'opaque bytes, deliberately not NPZ')
    manifest = dict(name='frozen',args=dict(K=2),splits={s:dict(sha256=sha(data/(s+'.npz'))) for s in splits})
    (data/'manifest.json').write_text(json.dumps(manifest))
    source = tmp_path/'source.py'; source.write_text('frozen source')
    admission = dict(kind='classical',data='frozen',fit_runs=2000,data_manifest_sha256=sha(data/'manifest.json'),
                     source_sha256={'source.py':sha(source)})
    assert verify_admission(admission,root=tmp_path)[0] is None
    source.write_text('changed')
    with pytest.raises(ValueError,match='source'): verify_admission(admission,root=tmp_path)
    source.write_text('frozen source'); (data/'test_clean.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='dataset bytes'): verify_admission(admission,root=tmp_path)


def test_queue_builder_waits_for_artifacts_then_binds_all_twelve_jobs(tmp_path,monkeypatch):
    from scripts import await_curie_fas_stage4 as watcher
    monkeypatch.setattr(watcher,'ROOT',tmp_path)
    data=tmp_path/'experiments/data/fas'/watcher.DATA;data.mkdir(parents=True)
    splits=('train_clean','val_clean','val_faulty','test_clean','test_faulty')
    for split in splits:(data/(split+'.npz')).write_bytes(b'opaque synthetic bytes')
    manifest=dict(name=watcher.DATA,args=dict(K=2),splits={s:dict(sha256=sha(data/(s+'.npz'))) for s in splits})
    (data/'manifest.json').write_text(json.dumps(manifest));data_hash=sha(data/'manifest.json')
    for path in watcher.TOOLS:
        source=tmp_path/path;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('synthetic source')
    source=tmp_path/'model.py';source.write_text('synthetic model')
    source_hash={'model.py':sha(source)}
    (tmp_path/'experiments/queue').mkdir(parents=True)
    assert watcher.prepare('fixture_scoring',dict(tag='contract'))[0] is None
    for kind,seeds in (('native',(6,7,8)),('lstm',(0,1,2)),('transformer',(0,1,2))):
        for seed in seeds:
            if kind=='native':
                tag=f'curie_b3_stage4_c10_s{seed}_20261009T0600Z'
                directory=tmp_path/'experiments/results/fas';cp=directory/(tag+'.pt')
                args=dict(pred_window=64,keyed=0,epochs=3,d=32,modes=16,layers=2,n_exp=2,n_lognormal=8,
                          n_window=4,dv=4,dropout=.1,lr=.003,batch=16,fit_runs=5000,
                          train_max_events=1024,max_events=1100,eval_runs=1000)
            else:
                stamp='20261006T1815Z' if seed==0 else '20261009T1330Z'
                tag=f'aws_fas_v2_ref_{kind}_d128_lr0.003_s{seed}_{stamp}'
                directory=tmp_path/'experiments/results/aws_20260929'/tag;cp=directory/'checkpoints'/'selected.pt'
                args=dict(model=kind,d=128,layers=2,lr=.003,no_test=True)
            directory.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(b'opaque weights')
            args.update(seed=seed,tag=tag,data=watcher.DATA)
            r=dict(status='completed',args=args,source_sha256=source_hash,data_manifest_sha256=data_hash,
                   test_auroc='not scored (development run)')
            if kind=='native':r['checkpoint']=str(cp.relative_to(tmp_path))
            else:r['selected_weights']=dict(path='checkpoints/selected.pt',sha256=sha(cp))
            (directory/(tag+'.json')).write_text(json.dumps(r))
    classic=tmp_path/'experiments/results/fas/fas_v2_classical_val_20261006T2115Z.json'
    classic.write_text(json.dumps(dict(status='completed',split='val',data=watcher.DATA,fit_runs=2000,
                                      source_sha256=source_hash)))
    prepared,missing=watcher.prepare('fixture_scoring',dict(tag='contract'))
    assert not missing
    plan=json.loads((tmp_path/prepared[0]).read_text())
    assert len(plan['jobs'])==12
    assert len(plan['jobs'][-1]['requires'])==11
    assert all(j['rss_kb']==3000000 and j['timeout_s']==7200 for j in plan['jobs'][1:-1])
    assert all(sha(tmp_path/j['queue'])==j['queue_sha256'] for j in plan['jobs'][1:])
    # Changed checkpoints are rejected by the actual scorer verifier before loading arrays.
    admission_path=next((tmp_path/'experiments/queue/fixture_scoring').glob('*native_s6_admission.json'))
    admission=json.loads(admission_path.read_text());(tmp_path/admission['checkpoint']).write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='checkpoint'):verify_admission(admission,root=tmp_path)
