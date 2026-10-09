"""B2 contract: v8/v9 forward parity and exact optimizer/EMA/RNG recovery."""
import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from race_irts_v8 import RaceIRTS as V8
from race_irts_v9 import RaceIRTS as V9
from recovery_state import load_state, save_state

ROOT = Path(__file__).resolve().parents[2]


def check():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    old = V8(2, 1, 4, 2, 1, 2, 2, 0.2, 3)
    new = V9(2, 1, 4, 2, 1, 2, 2, 0.2, 3)
    new.load_state_dict(old.state_dict())
    t = torch.arange(5).float().repeat(3, 1)
    z = torch.randn(3, 5, 2)
    mask = torch.ones_like(z)
    lengths = torch.full((3,), 5)
    static = torch.randn(3, 1)
    old.eval(); new.eval()
    parity = (old(t, z, mask, lengths, static) - new(t, z, mask, lengths, static)).abs().max().item()
    assert parity == 0
    optimizer = torch.optim.AdamW(new.parameters(), lr=.002)
    ema = torch.optim.swa_utils.AveragedModel(new, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999))
    random.seed(19); np.random.seed(23)

    def update(model, opt, avg):
        model.train()
        permutation = np.random.permutation(3)
        jitter = 1 + .1 * torch.randn(3, 1, 2)
        # Exercise Python RNG too, independently of the current PAM recipe.
        scale = .95 + .1 * random.random()
        loss = torch.nn.functional.cross_entropy(model(t[permutation], z[permutation] * jitter * scale,
                                                       mask[permutation], lengths[permutation], static[permutation]),
                                                 torch.tensor([0, 1, 2])[permutation])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); avg.update_parameters(model)
        return loss.item()

    update(new, optimizer, ema)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'state.pt'
        metadata = dict(protocol='B2 recovery contract', source='fixed', data='fixed')
        save_state(path, model=new, optimizer=optimizer, ema=ema, metadata=metadata, next_epoch=1)
        expected_loss = update(new, optimizer, ema)
        expected_model = {k: v.clone() for k, v in new.state_dict().items()}
        expected_ema = {k: v.clone() for k, v in ema.state_dict().items()}
        restored = V9(2, 1, 4, 2, 1, 2, 2, .2, 3)
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=.002)
        restored_ema = torch.optim.swa_utils.AveragedModel(restored, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.999))
        state = load_state(path, model=restored, optimizer=restored_opt, ema=restored_ema, metadata=metadata)
        assert state['next_epoch'] == 1
        actual_loss = update(restored, restored_opt, restored_ema)
        model_error = max((expected_model[k] - v).abs().max().item() for k, v in restored.state_dict().items())
        ema_error = max((expected_ema[k] - v).abs().max().item() for k, v in restored_ema.state_dict().items())
        assert expected_loss == actual_loss and model_error == 0 and ema_error == 0
        for original, recovered in zip(optimizer.state.values(), restored_opt.state.values()):
            assert all(torch.equal(original[k], recovered[k]) for k in original)
        try:
            load_state(path, model=restored, optimizer=restored_opt, ema=restored_ema,
                       metadata={**metadata, 'source': 'changed'})
        except ValueError:
            pass
        else:
            raise AssertionError('Changed source must refuse recovery')
    return dict(v8_v9_forward_max_error=parity, resumed_model_max_error=model_error,
                resumed_ema_max_error=ema_error, resumed_loss_error=abs(actual_loss - expected_loss),
                optimizer_exact=True, source_change_refused=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True); args = parser.parse_args()
    sources = [Path(__file__), *[Path(__file__).with_name(n) for n in
                               ('race_irts_v8.py', 'race_irts_v9.py', 'recovery_state.py')],
               ROOT / 'experiments/tpp/race_tpp_v8.py']
    result = dict(status='completed', battle='B2', tag=args.tag, checks=check(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    with (ROOT / 'experiments/results/irts' / f'{args.tag}.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result['checks']), flush=True)
