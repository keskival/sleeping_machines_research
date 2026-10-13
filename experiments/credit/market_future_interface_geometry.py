"""B10 native future-state credit acquisition from forward directional probes."""
import argparse, copy, json, time
from market_native_remote_admission_selected import (
    ROOT, torch, np, restored, load, warm, control, flat, sha, NativeMarketCredit)
from experiments.market.projected_clock_state import ProjectedClockState


def producer(model, state, t, m, start, group):
    trace = ProjectedClockState(model, state)
    scores = []
    for k in range(start, start+group):
        forecast, state = trace.advance(t[k:k+1], m[k:k+1], state, control.market_consume)
        scores.append(float(-control.revealed_log_probability(model, forecast, t[k+1:k+2], m[k+1:k+2]).mean().detach()))
        state = control.detach_state(state)
    return state, trace.sensitivity, scores


def future(model, state, t, m, start, group):
    scores = []
    for k in range(start, start+group):
        forecast, state = control.market_consume(model, t[k:k+1], m[k:k+1], state)
        scores.append(-control.revealed_log_probability(model, forecast, t[k+1:k+2], m[k+1:k+2]).mean())
    return torch.stack(scores).mean(), state


def shifted(state, sensitivity, direction, epsilon):
    state = copy.deepcopy(control.detach_state(state))
    for z, e in zip(control.tensors(state), sensitivity):
        z.add_(epsilon * torch.as_tensor(e[:, direction]).reshape_as(z))
    return state


def forward_labels(model, state, sensitivity, t, m, start, group, epsilon):
    values = []
    with torch.no_grad():
        for direction in range(4):
            plus, _ = future(model, shifted(state, sensitivity, direction, epsilon), t, m, start, group)
            minus, _ = future(model, shifted(state, sensitivity, direction, -epsilon), t, m, start, group)
            values.append(float((plus-minus)/(2*epsilon)))
    return np.array(values)


def contract(model, t, m, state, start):
    group = 8
    end, sensitivity, _ = producer(model, copy.deepcopy(state), t, m, start, group)
    boundary = copy.deepcopy(end)
    for z in control.tensors(boundary):
        z.requires_grad_()
    loss, _ = future(model, boundary, t, m, start+group, group)
    cotangents = torch.autograd.grad(loss, tuple(control.tensors(boundary)))
    expected = sum((e.T @ c.detach().numpy().reshape(-1) for e, c in zip(sensitivity, cotangents)), np.zeros(4))
    # Full producer + future graph; subtract parameter derivatives introduced
    # only in the future block to isolate the complete producer-state path.
    full = copy.deepcopy(state)
    for k in range(start, start+group):
        _, full = control.market_consume(model, t[k:k+1], m[k:k+1], full)
    total, _ = future(model, full, t, m, start+group, group)
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(total, parameters, allow_unused=True)
    local_loss, _ = future(model, copy.deepcopy(end), t, m, start+group, group)
    local = torch.autograd.grad(local_loss, parameters, allow_unused=True)
    by = {name: (torch.zeros_like(p) if a is None else a) - (torch.zeros_like(p) if b is None else b)
          for (name,p),a,b in zip(model.named_parameters(),gradients,local)}
    exact = np.array([float(by[f'layers.{j}.{key}'].sum())/np.sqrt(layer.n)
                      for j,layer in enumerate(model.layers) for key in ('log_rate','freq')])
    error = float(np.max(np.abs(exact-expected))); assert error < 1e-10
    fd = []
    for epsilon in (1e-3, 3e-4, 1e-4):
        labels = forward_labels(model, end, sensitivity, t, m, start+group, group, epsilon)
        fd.append(dict(epsilon=epsilon, max_error=float(np.max(np.abs(labels-expected)))))
    assert fd[-1]['max_error'] < 1e-6
    return dict(projected_state_vs_full_history_error=error, finite_difference=fd,
                complete_boundary_floats=sum(z.numel() for z in control.tensors(end)),
                eligibility_floats=sum(e.size for e in sensitivity), exact_directional_credit=expected.tolist())


def collect(model, critic, optimizer, t, m, state, blocks, training, scale):
    carry = None; records = []; group = 8; previous_credit = np.zeros(4)
    for block in range(blocks):
        k = 127 + block * 2 * group
        old = flat(state)
        end, sensitivity, scores = producer(model, state, t, m, k, group)
        current = flat(end)
        descriptor = (current-old).clone()
        eligibility = np.concatenate(sensitivity,0)
        # Issued operator geometry and already revealed credit, all causal.
        descriptor[:4] = torch.as_tensor(np.arcsinh(previous_credit / scale))
        descriptor[4:20] = torch.as_tensor(np.arcsinh(eligibility.T @ eligibility)).flatten()
        descriptor[20:24] = torch.as_tensor(np.arcsinh(eligibility.T @ current.numpy()))
        descriptor[24:28] = torch.as_tensor(np.arcsinh(eligibility.T @ (current-old).numpy()))
        features = torch.cat((old,current,descriptor))[None]
        feedback = current.new_zeros(8)
        feedback[0] = torch.log1p((t[k+group]-t[k])/model.scale)
        feedback[1:7] = torch.nn.functional.one_hot(m[k+1:k+group+1],6).double().mean(0)
        feedback[7] = float(np.mean(scores))
        events = torch.cat((features[0], feedback))[None,None]
        critic.core.train(training)
        with torch.set_grad_enabled(training):
            prediction, carry = critic(features, events, carry)
        # These observations become available only after the prediction above.
        target = forward_labels(model, end, sensitivity, t, m, k+group, group, 1e-4)
        target_tensor = torch.as_tensor(target / scale)[None]
        raw = prediction.detach().numpy()[0]*scale
        records.append(dict(target=target.tolist(), prediction=raw.tolist()))
        if training:
            optimizer.zero_grad(set_to_none=True)
            (prediction-target_tensor).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            _, state = future(model, end, t, m, k+group, group)
        state = control.detach_state(state)
        previous_credit = target
    target = np.array([r['target'] for r in records]); predicted = np.array([r['prediction'] for r in records])
    return dict(blocks=blocks, records=records, mse=float(np.mean((predicted-target)**2)),
                zero_credit_mse=float(np.mean(target**2)),
                forward_teacher_worlds=8*blocks, forward_teacher_targets=64*blocks)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag',required=True)
    parser.add_argument('--mode',choices=('contract','fit'),required=True); args=parser.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(19124)
    started=time.monotonic(); model, inputs = restored()
    train_blocks = 128 if args.mode == 'fit' else 1
    t,m,path = load('2026-08-24',128+16*train_blocks); inputs[str(path.relative_to(ROOT))]=sha(path)
    state = warm(model,t,m,128)
    checked = contract(model,t,m,state,127)
    result = dict(status='completed',battle='B10',args=vars(args),contract=checked)
    if args.mode == 'fit':
        critic=NativeMarketCredit(264).double()
        critic.core.head=torch.nn.Linear(critic.core.total_payload,4).double()
        torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
        optimizer=torch.optim.Adam(critic.parameters(),lr=.001)
        # Fixed scaling determined from first TRAIN diagnostic only.
        scale=np.maximum(1e-4,np.abs(np.array(checked['exact_directional_credit'])))
        result['scale']=scale.tolist()
        result['train']=collect(model,critic,optimizer,t,m,state,train_blocks,True,scale)
        td,md,path=load('2026-08-25',128+16*32);inputs[str(path.relative_to(ROOT))]=sha(path)
        result['dev']=collect(model,critic,optimizer,td,md,warm(model,td,md,128),32,False,scale)
        result['critic_parameters']=sum(p.numel() for p in critic.parameters())
    sources=('experiments/credit/market_future_interface_geometry.py','experiments/market/projected_clock_state.py',
             'experiments/credit/native_market_credit.py','experiments/market/episode_state.py')
    result.update(wall_s=time.monotonic()-started,input_sha256=inputs,
                  source_sha256={p:sha(ROOT/p) for p in sources},
                  scope='Operator-conditioned variant:16 eligibility Gram entries,8 current/change projections,4 previous revealed credits,per-direction TRAIN scaling; same native architecture/parameter count. One fixed restored market actor and native persistent race critic; complete two-layer plus addressed-state rank4 eligibility. TRAIN-only future directional finite-difference teaching; causal critic prediction before each future block. Exact BP used only for numerical contract, not fit labels. DEV repeated stream; no actor update/forecast-quality, complete-work, unknown-regime or hardware win claimed. Eight forward teacher worlds per block explicitly charged as targets; source actor original fitting additionally paid.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','dev')}),flush=True)
    for key in ('train','dev'):
        if key in result: print(key,json.dumps({k:v for k,v in result[key].items() if k!='records'}),flush=True)


if __name__ == '__main__':main()
