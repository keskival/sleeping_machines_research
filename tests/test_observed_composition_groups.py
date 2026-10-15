from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.paired_composition_decision import observed_groups


def test_uncertainty_groups_use_public_geometry_and_cover_correlated_views():
    rows=[r.model_input() for r in corpus() if r.split=='dev']
    groups=observed_groups(rows)
    assert len(groups)==16 and all(len(g)==4 for g in groups)
    assert sorted(i for group in groups for i in group)==list(range(64))


def test_fixed_child_delay_is_causal_and_keeps_timer_random_draws():
    import torch
    from sleeping_machines.frame_native_composition import FrameNativeComposition
    from experiments.hierarchy.paired_composition_decision import apply_emission_cut
    args=dict(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2)
    torch.manual_seed(11);base=FrameNativeComposition(**args).double()
    torch.manual_seed(11);cut=FrameNativeComposition(**args).double();apply_emission_cut(cut,'child_delays')
    states=[]
    for model in [base,cut]:
        state=model.new_state();model.observe(state,(0,0),0.,[1.,1.]);model.drain(state)
        states.append(state)
    torch.manual_seed(13);a=base.objects[0].arm(states[0].banks[0],(0,0),states[0].now);rng=torch.get_rng_state()
    torch.manual_seed(13);b=cut.objects[0].arm(states[1].banks[0],(0,0),states[1].now)
    assert torch.equal(rng,torch.get_rng_state())
    assert b.due>=a.due
    assert abs(float(b.due)-states[1].now-.011)<1e-12
    assert cut.objects[0].fire(states[1].banks[0],b,b.due) is not None


def test_joint_value_delay_cut_preserves_finite_native_execution():
    import torch
    from sleeping_machines.frame_native_composition import FrameNativeComposition
    from experiments.hierarchy.paired_composition_decision import apply_emission_cut
    torch.manual_seed(17)
    model=FrameNativeComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double().eval()
    apply_emission_cut(model,'values_and_child_delays');state=model.new_state()
    for i,xy in enumerate([(0,0),(0,100),(100,0),(100,100)]):model.observe(state,xy,i*.01,[(-1.)**i,1.])
    with torch.no_grad():output,time=model.compose(state)
    assert torch.isfinite(output).all() and torch.isfinite(time)
    assert not state.queue and sum(state.input_issued.values())==4
