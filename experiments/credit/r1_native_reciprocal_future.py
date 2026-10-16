"""Acquired eight-layer native reciprocal/future-credit integration gate.

Actual action and no-action branches continue their own ordinary learning.
The issued branch is committed before outcomes and is never retrospectively
replaced by the better audit branch. All data are chronological TRAIN.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import os
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments')]
import numpy as np
import torch
import pytest
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM
from experiments.native_query_migration import migrate
from experiments.r1_composition_credit_pair import source_migration
from sleeping_machines.producer_replay_native_race import enable_producer_credit
from sleeping_machines.document_boundary_causes import enable_document_relevance
from sleeping_machines.coupled_native_query_credit import enable_coupled_query_credit
from sleeping_machines.compositional_producer_credit import enable_compositional_credit
from sleeping_machines.sampled_compositional_credit import enable_sampled_compositional_credit
from sleeping_machines.deferred_episodic_writer_credit import enable_deferred_episodic_credit
from sleeping_machines.native_future_update_credit import NativeFutureUpdateCredit, addressed_mix_action
from tests.test_native_query_migration import identical


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_sources():
    sources = [str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]
    sources += ['experiments/credit/r1_native_reciprocal_future.py',
                'tests/test_deferred_episodic_writer_credit.py', 'tests/test_native_future_update_credit.py',
                'tests/test_native_query_migration.py', 'tests/test_native_indexed_cause_head.py',
                'experiments/native_query_migration.py', 'experiments/r1_composition_credit_pair.py',
                'experiments/r1_local_native_deep.py', 'experiments/r1_cerl.py']
    return {n:sha(ROOT/n) for n in sorted(set(sources))}


def actor(saved):
    model = LocalNativeDeepLM(50257, saved['model']['causes.class_of'], saved['model']['causes.class_words'],
                              width=64, depth=8, heads=4, pool=16, concept_units=4096,
                              topk=4, explore=2, dk=32, packed=True)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    model, opt, state, _ = migrate(model, opt, saved['state'])
    enable_producer_credit(model, state)
    enable_document_relevance(model, opt)
    model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
    state = saved['state']
    enable_coupled_query_credit(model, state)
    enable_compositional_credit(model, state)
    enable_sampled_compositional_credit(model, state)
    enable_deferred_episodic_credit(model, state)
    g = torch.Generator().set_state(saved['g']); a = torch.Generator().set_state(saved['alt'])
    torch.set_rng_state(saved['rng'])
    return model, opt, state, g, a


def dump(path, model, opt, state, g, a, cursor):
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), state=state,
                    g=g.get_state(), alt=a.get_state(), rng=torch.get_rng_state(), cursor=cursor), path)


def update(model, opt, state, g, a, data, cursor, targets=4):
    tokens = torch.from_numpy(np.asarray(data[cursor:cursor+targets+1]).astype(np.int64))
    model.train(); opt.zero_grad(set_to_none=True)
    nll, obj, _, packets, _ = model.stream_nll(tokens[:-1], tokens[1:], state, g, a)
    assert bool(torch.isfinite(obj))
    obj.backward(); model.deliver_credit(packets, state)
    mix_gradient = model.core.mix.grad.detach().clone()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    return nll.detach().double().tolist(), mix_gradient, float(norm)


def advance(model, opt, state, g, a, data, start, horizon):
    losses = []
    for cursor in range(start, start+horizon, 4):
        row, _, _ = update(model, opt, state, g, a, data, cursor)
        losses.extend(row)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--mode', choices=['contract', 'acquired', 'continue'], required=True)
    parser.add_argument('--parent')
    args = parser.parse_args()
    torch.set_num_threads(1); started = time.monotonic()
    past_runtime=0.; past_peak=0
    result_path=ROOT/f'experiments/results/diagnostics/{args.tag}.json'
    assert not result_path.exists()
    loaded_sources = runtime_sources()
    if args.mode != 'continue':
        assert pytest.main(['-q', 'tests/test_deferred_episodic_writer_credit.py',
                            'tests/test_native_future_update_credit.py']) == 0
    result = dict(status='completed', battle='R1', gate='PASS',
                  stage='native reciprocal future credit numerical gate', rows=[])
    if args.mode != 'contract':
        continuation = args.mode == 'continue'
        parent_result = json.loads((ROOT/args.parent).read_text()) if continuation else None
        if continuation:
            assert parent_result['status']=='completed' and parent_result['gate']=='PASS'
            assert parent_result['stage']=='acquired native reciprocal future credit integration gate'
            for name, digest in parent_result['source_sha256'].items():
                assert sha(ROOT/name)==digest, name
            parent_path = ROOT/parent_result['checkpoint']
            assert sha(parent_path)==parent_result['checkpoint_sha256']
        else:
            parent_path = ROOT/'experiments/results/token_language/aws_r1_native_document_credit_4m_s0_20261016_v1.resume.pt'
        # The active trainer atomically replaces this file; retain one exact read.
        raw = parent_path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        import io
        parent = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False); del raw
        migration = [] if continuation else source_migration(parent)
        cursor = parent['cursor']
        model, opt, state, g, a = actor(parent)
        data = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
        actor_rng = torch.get_rng_state()
        torch.manual_seed(19173)
        critic = NativeFutureUpdateCredit(64, depth=8, heads=4, pool=4)
        meta = torch.optim.AdamW(critic.parameters(), lr=.001)
        credit_state = critic.new_state()
        cg = torch.Generator().manual_seed(19174); ca = torch.Generator().manual_seed(19175)
        policy_rng = torch.Generator().manual_seed(19176)
        if continuation:
            critic.load_state_dict(parent['critic']); meta.load_state_dict(parent['meta'])
            credit_state=parent['credit_state']; cg.set_state(parent['cg']); ca.set_state(parent['ca'])
            policy_rng.set_state(parent['policy_rng'])
        torch.set_rng_state(actor_rng)
        del parent
        episodes, horizon = (32,32) if continuation else (2,8)
        rows = []; start_cursor = cursor
        progress = ROOT/f'experiments/results/credit/{args.tag}.resume.pt'
        with tempfile.TemporaryDirectory(prefix='reciprocal-native-', dir=ROOT/'.git') as folder:
            anchor = Path(folder)/'before_action.pt'; issued = Path(folder)/'issued.pt'
            baseline = Path(folder)/'continuous_baseline.pt'
            if continuation:
                dump(baseline, model, opt, state, g, a, cursor)
            if continuation and progress.exists():
                resumed = torch.load(progress, map_location='cpu', weights_only=False)
                assert resumed['source_sha256']==loaded_sources and resumed['parent_sha256']==digest
                del model, opt, state, g, a; gc.collect()
                model, opt, state, g, a = actor(resumed['actor'])
                cursor=resumed['actor']['cursor']; start_cursor=resumed['start_cursor']; rows=resumed['rows']
                critic.load_state_dict(resumed['critic']); meta.load_state_dict(resumed['meta'])
                credit_state=resumed['credit_state']; cg.set_state(resumed['cg']); ca.set_state(resumed['ca'])
                policy_rng.set_state(resumed['policy_rng']); torch.save(resumed['baseline'],baseline)
                past_runtime=resumed['recorded_process_s'];past_peak=resumed['max_rss_kb']
                del resumed
            for episode in range(len(rows), episodes):
                # Only already revealed support observations construct an action.
                support_losses, gradient, support_norm = update(model, opt, state, g, a, data, cursor, targets=4)
                cursor += 4
                all_quotes = []; all_packets = []; directions = []
                for depth in range(8):
                    key = state['publications'][depth][-1]
                    record = state['bindings'].entries[key].native
                    directions.append(addressed_mix_action(model, {'core.mix':gradient}, depth))
                    # Describe the actual action at this producer interface,
                    # rather than averaging parameter coordinates together.
                    summary = directions[depth] @ record.residual
                    quote, credit_state, packets = critic.quote(
                        record.key_view, record.residual, summary, depth/7, .001,
                        credit_state, cg, ca)
                    all_quotes.append(quote); all_packets.extend(packets)
                quotes = torch.stack(all_quotes).flatten()
                selected, probability = critic.choose(quotes, policy_rng)
                if episode == 0 and not continuation:
                    # A declared bootstrap audit, selected before future data.
                    selected, probability = 7*3, 1.
                depth, action_index = divmod(selected, 3)
                sign = action_index-1
                predictions = quotes.detach().tolist()
                dump(anchor, model, opt, state, g, a, cursor)
                before = model.core.mix[depth].detach().clone()
                with torch.no_grad():
                    model.core.mix[depth].add_(sign*directions[depth])
                change = float((model.core.mix[depth]-before).norm())
                # This is the issued trajectory even if the later audit is worse.
                actual_losses = advance(model, opt, state, g, a, data, cursor, horizon)
                issued_writer_replays = state['episodic_writer_replays']
                dump(issued, model, opt, state, g, a, cursor+horizon)
                del model, opt, state, g, a; gc.collect()
                saved = torch.load(anchor, map_location='cpu', weights_only=False)
                model, opt, state, g, a = actor(saved)
                assert identical((model.state_dict(),opt.state_dict(),state,g.get_state(),a.get_state(),torch.get_rng_state()),
                                 (saved['model'],saved['optimizer'],saved['state'],saved['g'],saved['alt'],saved['rng']))
                del saved
                reference_losses = advance(model, opt, state, g, a, data, cursor, horizon)
                assert quotes.detach().tolist() == predictions, 'future audit changed an issued quote'
                advantage = (sum(actual_losses)-sum(reference_losses))/horizon
                teaching = critic.learn_revealed(quotes[selected], advantage, all_packets, credit_state, meta)
                rows.append(dict(episode=episode, issued_depth=depth, issued_sign=sign,
                                 selection='precommitted bootstrap' if episode==0 and not continuation else 'native critic policy',
                                 action_probability=probability, parameter_change_norm=change,
                                 issued_future_nll=sum(actual_losses)/horizon,
                                 audit_no_action_future_nll=sum(reference_losses)/horizon,
                                 issued_losses=actual_losses, audit_losses=reference_losses,
                                 support_losses=support_losses,
                                 actor_support_preclip_norm=support_norm,
                                 issued_writer_replays=issued_writer_replays, **teaching))
                del model, opt, state, g, a; gc.collect()
                if continuation:
                    saved = torch.load(baseline, map_location='cpu', weights_only=False)
                    assert saved['cursor']==cursor-4
                    model, opt, state, g, a = actor(saved); del saved
                    baseline_losses = advance(model, opt, state, g, a, data, cursor-4, horizon+4)
                    dump(baseline, model, opt, state, g, a, cursor+horizon)
                    rows[-1]['continuous_baseline_losses'] = baseline_losses
                    rows[-1]['continuous_baseline_nll'] = sum(baseline_losses)/(horizon+4)
                    del model, opt, state, g, a; gc.collect()
                # Never select the audit's better endpoint retrospectively.
                saved = torch.load(issued, map_location='cpu', weights_only=False)
                model, opt, state, g, a = actor(saved); cursor = saved['cursor']; del saved
                if continuation:
                    # One atomic package binds both causal learners and the
                    # credit learner at a completed update boundary.
                    actor_state=dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,
                                     g=g.get_state(),alt=a.get_state(),rng=torch.get_rng_state(),cursor=cursor)
                    bundle=dict(actor=actor_state,baseline=torch.load(baseline,map_location='cpu',weights_only=False),
                                critic=critic.state_dict(),meta=meta.state_dict(),credit_state=credit_state,
                                cg=cg.get_state(),ca=ca.get_state(),policy_rng=policy_rng.get_state(),
                                rows=rows,start_cursor=start_cursor,parent_sha256=digest,source_sha256=loaded_sources,
                                recorded_process_s=past_runtime+time.monotonic()-started,
                                max_rss_kb=max(past_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                    writing=progress.with_suffix('.writing');torch.save(bundle,writing);os.replace(writing,progress)
                    del bundle,actor_state
                print(json.dumps(dict(episode=episode,cursor=cursor,advantage=advantage,
                                      squared_error=teaching['squared_error'])),flush=True)
            if continuation:
                baseline_final=ROOT/f'experiments/results/credit/{args.tag}.baseline.pt'
                if baseline_final.exists():
                    assert sha(baseline_final)==sha(baseline)
                else:
                    shutil.copyfile(baseline,baseline_final)
        if not continuation:
            assert rows[0]['parameter_change_norm'] > 0
        assert all(r['issued_writer_replays'] > 0 for r in rows)
        result.update(stage='native reciprocal future credit prospective continuation' if continuation else 'acquired native reciprocal future credit integration gate', rows=rows,
                      parent_sha256=digest, parent_cursor=start_cursor, source_migration=migration,
                      actual_train_targets=episodes*(horizon+4), additional_counterfactual_targets=episodes*horizon,
                      additional_continuous_baseline_targets=episodes*(horizon+4) if continuation else 0,
                      forward_depth=8, credit_depth=8,
                      credit_parameters=sum(p.numel() for p in critic.parameters()),
                      actor_parameters=sum(p.numel() for p in model.parameters()),
                      final_cursor=cursor,
                      scope='Two chronological TRAIN integration episodes, not a predictive win. Actor retains ordinary AdamW, numerical memory, indexed causes, temporal races, joint queries, full-depth sampled producer credit and deferred episodic writer credit. Credit machine is an eight-layer native persistent race model. Actions are one addressed mix-block step with both signs and no-op; other interventions remain outside this gate. Actual finite future branches include subsequent actor state and ordinary learning. No future labels enter issued action selection, no hindsight rollback, TEST, dense control or hardware/work win. Conditional eligibility and changing-parameter replay remain approximate; action-value training is not an exact whole-history meta-gradient.')
        if continuation:
            result.pop('gate',None)
            issued_losses=[v for row in rows for v in row['support_losses']+row['issued_losses']]
            baseline_losses=[v for row in rows for v in row['continuous_baseline_losses']]
            result.update(scope='One acquired parent, one chronological TRAIN seed and 32 prospective episodes. Two complete native learning pipelines share initial parameters, Adam and numerical state. The candidate adds a native future-valued update policy and paid controlled audits; the reference continues its own ordinary causal learning and persistent state. Positive prequential gain favors the reciprocal learner. No TEST, public benchmark, iso-compute, hardware or arbitrary-action/full-history-exact claim. Actual action consequences train the critic after outcomes; the issued branch is never replaced retrospectively.',
                          baseline_checkpoint=str(baseline_final.relative_to(ROOT)),baseline_checkpoint_sha256=sha(baseline_final),
                          prequential_nll=sum(issued_losses)/len(issued_losses),
                          continuous_baseline_nll=sum(baseline_losses)/len(baseline_losses),
                          prequential_nll_gain=(sum(baseline_losses)-sum(issued_losses))/len(issued_losses),
                          mean_audited_action_advantage=sum(row['target_advantage'] for row in rows)/episodes,
                          critic_mean_squared_error=sum(row['squared_error'] for row in rows)/episodes,
                          zero_predictor_mean_squared_error=sum(row['target_advantage']**2 for row in rows)/episodes)
    result.update(source_sha256=loaded_sources,
                  elapsed_s=past_runtime+time.monotonic()-started,
                  process_scope='Recorded intervals including audit, actor and credit learning; unsaved interruption tails excluded. No process-cost win claimed.',
                  max_rss_kb=max(past_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if args.mode != 'contract':
        checkpoint = ROOT/f'experiments/results/credit/{args.tag}.pt'
        if checkpoint.exists():
            old_checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert old_checkpoint['source_sha256']==loaded_sources and old_checkpoint['parent_sha256']==digest
            del old_checkpoint
        writing=checkpoint.with_suffix('.writing')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), state=state,
                        g=g.get_state(), alt=a.get_state(), rng=torch.get_rng_state(), cursor=cursor,
                        critic=critic.state_dict(), meta=meta.state_dict(), credit_state=credit_state,
                        cg=cg.get_state(), ca=ca.get_state(), policy_rng=policy_rng.get_state(),
                        parent_sha256=digest, source_sha256=result['source_sha256']), writing)
        os.replace(writing,checkpoint)
        result.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint))
    result['elapsed_s']=past_runtime+time.monotonic()-started
    result['max_rss_kb']=max(past_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    writing=result_path.with_suffix('.writing')
    with writing.open('w') as f:
        json.dump(result, f, indent=2); f.write('\n')
    os.replace(writing,result_path)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
