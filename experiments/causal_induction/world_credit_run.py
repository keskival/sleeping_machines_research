"""Issued-before-feedback native learner with outcome/credit work receipts."""
import math
import time
import numpy as np
import torch
from public_stream import records
from task_taught_run import clip


def run(model, path, arm, optimizer=None, seed=71, max_queries=None):
    st = model.fresh()
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    scores, world_scores, labels, probabilities, losses = [], [], [], [], []
    trace = []
    begin = time.monotonic()
    updates = peak_bytes = 0
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for r in records(path):
            if r['kind'] == 'event':
                model.observe(r, st)
            elif r['kind'] == 'query':
                model.issue(r, st, generator, arm, optimizer is not None)
            else:
                packet = st['issued'][r['query_id']]
                world_probability = packet['world_probability']
                loss, probability = model.feedback(r, st, optimizer is not None)
                y = r['target']
                scores.append(-math.log(max(1e-12, probability if y else 1-probability)))
                world_scores.append(-math.log(max(1e-12, world_probability if y else 1-world_probability)))
                labels.append(y)
                probabilities.append(world_probability)
                trace.append(dict(issued_time=packet['time'], feedback_time=r['time'],
                                  query_id=r['query_id'], probability=probability,
                                  world_probability=world_probability, target=y))
                if optimizer is not None:
                    losses.append(loss)
                    if len(losses) >= 8 and not st['issued']:
                        optimizer.zero_grad()
                        torch.stack(losses).mean().backward()
                        clip(model)
                        optimizer.step()
                        losses = []
                        model.detach(st)
                        updates += 1
                        states = list(st['nodes'].values())+list(st['credit_nodes'].values())
                        state_bytes = sum(s.storage()['persistent_tensor_bytes'] for s in states)
                        bank_bytes = sum(r['value'].numel()*r['value'].element_size()
                                         for versions in st['bank'].values() for r in versions)
                        peak_bytes = max(peak_bytes, state_bytes+bank_bytes)
                if max_queries is not None and len(scores) >= max_queries:
                    assert not st['issued']
                    break
        assert not st['issued']
        if losses:
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            clip(model)
            optimizer.step()
            model.detach(st)
            updates += 1
    half = len(scores)//2
    assert scores and all(math.isfinite(x) for x in scores+world_scores)
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return dict(nll=float(np.mean(scores)), targets=len(scores),
                first_half_nll=float(np.mean(scores[:half])),
                second_half_nll=float(np.mean(scores[half:])),
                world_nll=float(np.mean(world_scores)),
                world_brier=float(np.mean((np.array(probabilities)-np.array(labels))**2)),
                world_first_half_nll=float(np.mean(world_scores[:half])),
                world_second_half_nll=float(np.mean(world_scores[half:])),
                positive_fraction=float(np.mean(labels)),
                wall_s=time.monotonic()-begin, optimizer_updates=updates,
                peak_persistent_tensor_bytes=peak_bytes,
                peak_occupied_nodes=st['peak_nodes'],
                peak_world_addresses=st['peak_credit_pairs'],
                issued_trace=trace, **st['counts'])
