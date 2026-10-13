"""B10 eligibility-derived final credit packets and reordered Adam bank delivery."""
import argparse,copy,json
from market_native_early_request import ROOT,torch,restored,load,warm,window,control,sha
from sleeping_machines.learning.local_credit_banks import LocalCreditBanks,CreditPacket


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    original,inputs=restored();t,m,path=load('2026-08-24',128+3*8)
    inputs[str(path.relative_to(ROOT))]=sha(path);rows=[]
    for arm in ['own','projected']:
        model=copy.deepcopy(original);reference=copy.deepcopy(original)
        optimizer=torch.optim.Adam(reference.parameters(),lr=.0003)
        banks=LocalCreditBanks(model.named_parameters(),lr=.0003,weight_decay=0.)
        state=warm(model,t,m,128);errors=[]
        for block in range(3):
            state,grads,scores=window(model,state,t,m,127+8*block,8,arm)
            optimizer.zero_grad(set_to_none=True)
            for p,g in zip(reference.parameters(),grads):p.grad=g.clone()
            # Match the admitted rule: global norm clipping is still charged.
            torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,error_if_nonfinite=True)
            packets=[CreditPacket(name,banks.versions[name],p.grad.detach().clone(),f'{arm}:{block}')
                     for name,p in reference.named_parameters()]
            optimizer.step()
            # Final gradients already issued against a common parameter version;
            # bank application may reorder, but prediction waits for this window.
            for packet in reversed(packets):banks.apply(packet)
            error=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),reference.parameters()))
            assert error<1e-12;errors.append(error)
        stale=False
        try:banks.apply(packets[0])
        except ValueError as exc:stale='Stale' in str(exc)
        assert stale
        rows.append(dict(credit=arm,updates=3,parameter_max_errors=errors,
            bank_packets=len(banks.receipts),gradient_payload_bytes=sum(r['gradient_bytes'] for r in banks.receipts),
            stale_packet_rejected=stale,storage=banks.storage(),global_clip_calls=3,
            within_window_history_backprop=False,bank_application_order='reverse named parameter order'))
    files=['experiments/credit/market_local_bank_contract.py',
           'sleeping_machines/learning/local_credit_banks.py',
           'experiments/credit/market_streaming_clock_updates.py',
           'experiments/market/local_eligibility_state.py','experiments/market/projected_clock_state.py']
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,
        source_sha256={p:sha(ROOT/p) for p in files},
        scope='Real TRAIN market actor gradients from own/projected temporal eligibility, no window-history graph. Three8target updates per endpoint, fresh Adam states; AdamW bank weight_decay0. Global norm clipping and common parameter/window issuance remain. Reordered bank application matches ordinary Adam after all packets delivered; strict stale rejection. Forecast still waits for completed parameter version. This is credit/update compatibility, not a fully asynchronous optimizer, local clipping policy, quality or hardware efficiency win.')
    out=ROOT/f'experiments/results/market/{args.tag}.json';assert not out.exists()
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
