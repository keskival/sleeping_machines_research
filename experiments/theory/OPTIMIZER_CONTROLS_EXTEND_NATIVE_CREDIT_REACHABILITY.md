# Optimizer controls extend native credit reachability

Curie,12October2026. B10. Concrete extension selected after the longer native market development stage and a source-bound derivative contract.

## The missing response directions are in the interface

The current prospective field changes the actor's parameter signal by J^T b, where J is the derivative of issued temporal producer state. In our market actor, clock.weight/bias, mark_ctx.weight/bias and mark_slot affect the prediction readout but do not produce that temporal state. Their columns in J are zero. Those3756/18164parameters receive ordinary exact backprop; the learned state field cannot directly change their current update through this J. Different producer updates can still affect their later gradients, so this is not a no-learning or family limitation.

The source contract verifies zero producer VJPs for all five readout tensors and current-loss gradient squared norm9.398731 on those tensors. It also verifies actual Adam proposal parity6.38e-13. This interface restriction follows both from the program's dependency graph and from its measured derivatives. It does not by itself prove the cause of the longer-stream loss.

## A connected critic can control credit and update geometry together

Let d(g,m) be the clipped, moment-normalized Adam displacement at the base learning rate, and g=g_exact+J^T b. Extend the update to

    theta'=theta-alpha d(g,m),
    alpha=exp(log(4)*tanh(r_phi)).

Both b and r_phi are outputs of the same native temporal/race model. The exact actor signal remains, and the new scalar uses the model's shared event memory and forward-response information. At zero r, alpha=1; the declared¼–4 trust interval keeps updates positive and bounded. This adds17readout parameters, with no new independent critic or dense recurrent module.

For a known future loss and fixed issued actor/moment state,

    partial L_future / partial log(alpha)
      = -alpha grad_theta L_future^T d.

The scalar teacher summarizes how the whole proposed update aligns with future loss, including readout directions absent from J^T b. Conditional information and the chosen control family still limit what a trained model can learn. A single scale cannot represent arbitrary per-parameter changes. Gradient clipping, moment updates and the actual proposal are charged and differentiated as implemented.

The completed source contract computes the actual alpha derivative0.0204729485, an exactly matching update-response contraction, and central finite differences with minimum error3.81e-9. At this source history, its positive sign favours reducing the step. This is a local utility result; it is not evidence that a trained rate controller has already improved forecasting.

## Evidence-driven experiment

The unchanged16384target prospective-field run improves versus frozen guidance+0.001801nats/event but trails online BPTT0.002338. Its first4096prefix exactly replays the earlier tiny win; the later quarters expose a sustained gap. The selected revision therefore adds useful actor-update directions rather than scaling the same restricted field or discarding the exact signal.

NATIVE_OPTIMIZER_FUTURE_PROTOCOL.md tests joint field/step learning against ordinary online BPTT, fully frozen, genuine local-only, frozen joint guidance, and online field with a fixed step. The fixed-step online arm uses the same acquired bundle, isolating the new control. Every forecast is scored before an outcome-dependent update can affect subsequent forecasts. No new competitor architecture, sealed test or trader-identity supervision is used.

The construction uses known optimizer-control and differentiation primitives. Its research question is the complete event-native, history-conditioned, future-taught learning system under measured resource constraints; familiar primitives neither establish nor refute its utility.
