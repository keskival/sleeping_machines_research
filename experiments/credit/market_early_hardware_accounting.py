"""Source-bound logical remote-bank arithmetic/access accounting; no training."""
import argparse, json
from market_native_early_request import ROOT, restored, torch, sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag', required=True); args=ap.parse_args()
    torch.set_num_threads(1); model, _=restored()
    parent=ROOT/'experiments/results/market/curie_b10_early_request_20261013_v1_fit.json'
    record=json.loads(parent.read_text()); assert record['status']=='completed'
    for name,digest in record['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    rank=2*len(model.layers)
    state_values=sum(2*l.n for l in model.layers)+model.K*model.marks.dv
    elements=state_values*rank
    maps=[]
    for j,l in enumerate(model.layers):
        for name,module in [('write',l.write),('gate',l.gate),('read',l.read),
                            ('mlp0',l.mlp[0]),('mlp2',l.mlp[2])]:
            maps.append(dict(name=f'layer{j}.{name}',rows=module.weight.shape[0],
                             columns=module.weight.shape[1]))
    maps.append(dict(name='marks.value',rows=model.marks.value.weight.shape[0],
                     columns=model.marks.value.weight.shape[1]))
    # Every listed NumPy matrix-times-rank-tangent occurs once in advance.
    macs=rank*sum(x['rows']*x['columns'] for x in maps)
    rows=[]
    for row in record['rows']:
        remote_events=row['remote_windows']*32
        calls=512 if row['arm'] in ['frozen','online'] else 0
        teachers=row['teachers']
        rows.append(dict(arm=row['arm'],nll=row['nll'],scored_targets=row['targets'],
                         remote_events=remote_events,
                         remote_matrix_flops=2*macs*remote_events,
                         remote_matrix_flops_per_scored_target=2*macs*remote_events/row['targets'],
                         remote_bank_read_write_values_lower_bound=2*elements*remote_events,
                         numpy_bank_access_values=5*elements*remote_events,
                         native_scored_keys=16*calls,native_selected_writes=8*calls,
                         native_value_proposals=8*calls+8*teachers,
                         extra_future_teacher_scored_targets=64*teachers,
                         actor_adam_updates=512,native_adam_updates=teachers))
    source=record['acquisition']; source_remote=source['remote_windows']*32
    result=dict(status='completed_logical_accounting',battle='B10',rows=rows,
                rank=rank,remote_bank_values=elements,previous_bank_values=elements,
                matrix_maps=maps,matrix_macs_per_remote_event=macs,
                common_acquisition_remote_matrix_flops=2*macs*source_remote,
                scalar_bytes_conventions=[1,2,4,8],
                synchronization=dict(global_periodic_clock_required=False,
                    native_local_joins='Head arrivals joined per layer; prior source context readiness joined locally.',
                    current_optimizer_barrier='32-target mean gradient, global norm clipping and shared Adam parameter-version boundary.',
                    implication='Remote bank activity is conditional and localizable; prototype still has window-level reductions. An entirely clockless distributed optimizer is not implemented.'),
                exclusions=['own-trace and actor arithmetic','native controller arithmetic and history-access traffic',
                            'remote elementwise arithmetic, reductions and special functions',
                            'additional old-state cotangent arithmetic','proposal and future-teacher arithmetic',
                            'parameter reads and cache hierarchy','optimizer arithmetic and traffic'],
                scope='Exact shape/call-count remote-matrix arithmetic subtotal, two FLOPs/MAC. NumPy accesses count copying sensitivity to previous (2S), reading previous for advance (S), writing new (S), reading previous for historical projection (S); excludes temporaries. Fused lower bound2S read/update-write values, not DRAM measurements. Multiply logical values by declared scalar bytes, not assumed hardware precision. All-key scoring remains charged; current AddressedEventHeads evaluates key_read for each scored candidate and does not use the other evaluator cached-key optimization. No complete-work, energy, ASIC-speed or fully clockless training win follows from this subtotal.',
                input_sha256={str(parent.relative_to(ROOT)):sha(parent)},
                source_sha256={p:sha(ROOT/p) for p in ['experiments/credit/market_early_hardware_accounting.py',
                    'experiments/market/projected_clock_state.py','sleeping_machines/addressed_event_heads.py']})
    out=ROOT/f'experiments/results/market/{args.tag}.json'; assert not out.exists()
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
