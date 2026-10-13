"""Frozen TRAIN-only integration check for forward route credit across windows."""
import torch


@torch.no_grad()
def check(model,args,driver,train):
    length=args.chunk
    if length<=args.credit_window:
        raise ValueError('Cross-window contract requires a longer optimizer chunk')
    ids=driver.interval_tensor(train,0,args.lanes*(length+1),args.lanes)
    inputs,labels=ids[:,:-1],ids[:,1:]
    token=args.credit_window-1;layer=0;head=0
    old=(model.force_site,model.suppress_site,model.shadow_mode,model.replay_window)
    old_mode=model.training
    try:
        model.eval();model.force_site=None;model.suppress_site=(token,layer,head);model.shadow_mode=False
        gen=torch.Generator().manual_seed(args.seed+700000)
        tape=gen.get_state().clone()
        factual,state,_=model(inputs,None,gen,args)
        winner=state['race_winners'][token,layer,:,head]
        model.shadow_mode=True
        model.force_site=(token,layer,head,winner)
        null_gen=torch.Generator();null_gen.set_state(tape)
        null,null_state,_=model(inputs,None,null_gen,args)
        assert torch.equal(null,factual),'Null forced winner changed predictions'
        assert driver.learner_fingerprint(null_state)==driver.learner_fingerprint(state),'Null replay changed numeric state'
        assert torch.equal(null_gen.get_state(),gen.get_state()),'Null replay changed tape consumption'
        model.force_site=(token,layer,head,(winner+1)%args.pool)
        alt_gen=torch.Generator();alt_gen.set_state(tape)
        alternative,_,_=model(inputs,None,alt_gen,args)
        assert torch.equal(alternative[:,:token],factual[:,:token]),'Future intervention changed earlier features'
        factual_losses=model.readout.nll(factual,labels).reshape(args.lanes,length)
        alternative_losses=model.readout.nll(alternative,labels).reshape(args.lanes,length)
        delta=alternative_losses-factual_losses
        assert bool(torch.isfinite(delta).all())
        after=delta[:,args.credit_window:]
        return dict(status='passed',site=[token,layer,head],null_feature_state_tape_parity=True,
            causal_prefix_parity=True,forward_targets=3*args.lanes*length,
            scored_targets=2*args.lanes*length,after_boundary_targets=after.numel(),
            mean_after_boundary_loss_delta=float(after.mean()),
            maximum_after_boundary_feature_change=float((alternative-factual)[:,args.credit_window:].abs().max()),
            scope='Frozen TRAIN prefix, first-time-held forced receiver intervention; actual downstream races and numerical memory replayed across detached factual windows. All future token inputs are observed causal inputs during continuation. No learned credit fit, quality gain or derivative of the full stochastic routing distribution claimed.')
    finally:
        model.force_site,model.suppress_site,model.shadow_mode,model.replay_window=old
        model.train(old_mode)
