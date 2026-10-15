"""Bind completed language depth and native composition development evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=[f'experiments/results/token_language/aws_r1_deep_keyed_L{d}_1M_s0_20261015_v3.json' for d in (2,4,8)]
    results=[json.loads((ROOT/p).read_text()) for p in paths]
    for r in results: assert r['status']=='completed' and r['score_65528']['targets']==65528
    for k in ['seed','train_tokens','lane','d','modes','heads','dk','dropout','lr','wd','batch']:
        assert all(r['args'][k]==results[0]['args'][k] for r in results),k
    prefix='experiments/results/hierarchy/'
    names=['aws_g_hier_native_composition_20261015_v1_data.json','aws_g_hier_native_composition_20261015_v1_learn.json','aws_g_hier_native_query_attribution_20261015_v1.json','aws_g_hier_native_frozen_readout_20261015_v1.json','native_composition_credit_contracts_20261015_v1.json']
    native=[json.loads((ROOT/prefix/n).read_text()) for n in names]
    assert all(r['status']=='completed' for r in native[:4])
    assert native[4]['status']=='passed' and native[4]['tests']==60
    rows=[{k:v for k,v in r.items() if k!='work'} for r in native[3]['rows']]
    packet=dict(status='completed',language_depth=dict(evidence_level='single-seed within-family development',targets=65528,
        unique_training_tokens=results[0]['args']['train_tokens'],seed=0,
        rows=[dict(layers=r['args']['layers'],nll=r['score_65528']['nll'],parameters=r['parameters'],epochs=r['epochs_run']) for r in results],
        nll_improvement=results[0]['score_65528']['nll']-results[-1]['score_65528']['nll'],
        scope='Same scored slice and unique training corpus; different epochs and total work. Quality gain only; no matched-compute or semantic-hierarchy claim.'),
        hierarchical_memory=dict(status='native FIT learnability demonstrated',abstraction_levels=8,refinements_per_level=4,cells=32,passed_contracts=60,
        parameters=169675,episodes=12,raw_events_per_episode=32,training_updates=64,noise_tapes=2,rows=rows,
        capability='Persistent native race objects compose child events and typed responses from lower learned memories, then generate a race-timed readout.',
        scope='One FIT recording. Prior, centre and scalar gain fitted/scored on same FIT logits. No held-out semantic transfer or benchmark claim.',
        next_gate='Held-out combinations of familiar parts, training-only calibration, memory-content and relative-pose interventions.',
        timer_credit='Forward-identical winner-averaged pure-timer interior credit implemented; no measured training-quality gain. Native value-routing credit unchanged.'),
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [*paths,*[prefix+n for n in names]]})
    engineering_names=['aws_g_hier_coalesced_preflight_20261015_v2.json',
        'aws_g_hier_frame_preflight_20261015_v1.json','aws_g_hier_coalesced_learning_20261015_v1.json',
        'aws_g_hier_coalesced_readout_20261015_v1.json','aws_g_hier_nuisance_factorial_20261015_v1.json',
        'aws_g_hier_coalesced_transfer_diagnosis_20261015_v1.json','native_frame_contracts_20261015_v1.json']
    engineering=[json.loads((ROOT/prefix/n).read_text()) for n in engineering_names]
    assert all(r['status']=='completed' for r in engineering)
    old,new=engineering[0]['measurements'];frame=engineering[1]
    assert frame['exact_translation_feature_equality'] and engineering[-1]['tests']==76
    packet['hierarchical_memory']['passed_contracts']=76
    packet['hierarchical_memory']['contract_scope']='Combined hierarchy/task/frame/timing/readout suite; original FIT probe had60 contracts.'
    packet['composition_engineering']=dict(status='measured mechanism and resource progress',
        coalesced_views=dict(baseline=old['work'],coalesced=new['work'],
            scope='Same32-event TRAIN scene and acquired parameters; output-square gradient check, not equal-quality comparison.'),
        observed_frame=dict(parameters=frame['parameters'],translation_feature_displacement=frame['feature_displacement']['translation_only'],
            combined_nuisance_before=frame['reference_final']['feature_displacement']['both'],
            combined_nuisance_after=frame['feature_displacement']['both'],work=frame['work_backward'],
            scope='Same acquired weights/factorial TRAIN scenes. World pixel memories and scene-relative upper ownership; no new parameters or sensor replay. Exact translation contract, not learned semantic transfer.'),
        completed_transfer_attempt=dict(dev_nll=engineering[2]['final']['dev']['calibrated_nll'],
            accuracy=engineering[2]['final']['dev']['accuracy'],prior_nll=0.6931471805599453,
            scope='First coalesced128-update synthetic whole-label fit failed unseen-combination DEV; frozen affine readout did not recover transfer. Recorded separately from native FIT success.'),
        next_gate='Repaired observed-frame128-update run, TRAIN-only calibration, cached features and unseen-combination DEV; TEST not model-scored.',
        accounting='Native candidate and message counts plus extra timer work; changed information paths, no matched-quality, total-FLOP, hardware energy or latency claim.')
    packet['input_sha256'].update({prefix+n:hashlib.sha256((ROOT/prefix/n).read_bytes()).hexdigest() for n in engineering_names})
    repaired_path=prefix+'aws_g_hier_frame_learning_20261015_v1.json'
    repaired=json.loads((ROOT/repaired_path).read_text());assert repaired['status']=='completed'
    packet['composition_engineering']['repaired_transfer_attempt']=dict(
        dev_raw_nll=repaired['final']['dev']['raw_nll'],
        dev_calibrated_nll=repaired['final']['dev']['calibrated_nll'],
        accuracy=repaired['final']['dev']['accuracy'],
        convex_readout_diagnostics=repaired['convex_readout_diagnostics'],
        scope='128whole-label TRAIN updates,same synthetic splits/order/optimizer,TRAIN-only calibration. No held-out transfer gain; TEST not model-scored.')
    packet['composition_engineering']['next_gate']='Reuse cached features to diagnose part/role binding and credit with translation controlled. No larger unchanged fit or depth sweep.'
    packet['input_sha256'][repaired_path]=hashlib.sha256((ROOT/repaired_path).read_bytes()).hexdigest()
    (ROOT/'investment/hierarchy_development_evidence.json').write_text(json.dumps(packet,indent=2)+'\n')
    print('Bound three language depths, five frozen native variants, engineering gates and 76 contracts.')
if __name__=='__main__':main()
