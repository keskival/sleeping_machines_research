# Delayed finite learner consequences for native credit

13 October2026, B10/R1 direction. Learned credit optimizes the causal learner, not just an issued-gradient approximation.

Let x include actor parameters, Adam moments, event memory and native critic numerical state. An issued normalized action a induces learner transition F(x,a,y). Later gradients and critic actions are recomputed from each resulting x. Define U_H(x,a;future) as the mean prequential loss of Hfuture outcomes under this closed-loop learner. Current and alternative worlds share observations because the forecasting model does not influence the recorded market stream. This is an intervention on the predictor's learner, not on market participants.

A finite receipt compares U_H(x,a) with U_H(x,a+delta). It includes future own-gradient responses, optimizer moments and native policy feedback without needing to differentiate those paths. The current implementation uses two actor updates/24postaction targets; future policy weights are fixed at receipt issue. Native current activations are trained after receipt maturity; full historical critic-write differentiation and real-world action feedback are separate capabilities. Clone/version boundaries and all teacher work are explicit.

## Preference labels versus loss-value labels

For an isotropic sphere delta=r u in d action dimensions and a smooth fixed-future utility gradient q, the small-radius binary preference update has expectation proportional to −q/||q||. With uncertain futures this averages normalized gradients rather than expected gradients; it can prefer frequent tiny wins over rare large losses. Equal-weight winner imitation is therefore not generally aligned with expected forecast NLL.

Positive utility-gain weighting repairs the leading-order signal. Put g(delta)=U(a)−U(a+delta). Symmetry gives

    E_delta[delta max(−q^T delta,0)] = −r² q/(2d).

If U has L-Lipschitz gradient on the action ball, then

    ||E_delta[delta max(g(delta),0)] + r²q/(2d)|| ≤ L r³/2.

Thus −2d delta max(g,0)/r² estimates q with norm bias at most d L r. A gain threshold tau adds at most2d tau/r to this bound. Conditioning on issued history permits uncertain future outcomes and a predictable positive scale; averaging the leading signal targets expected future demand. No stationarity assumption is needed, but changing conditional demands still require causal adaptation. Hard-route boundaries or Adam sign changes can violate smooth-region assumptions; finite observed consequences remain valid comparisons, while this gradient interpretation then needs additional care.

Train native current activations toward the checked candidate with gain-weighted squared error. Holding the target a+delta fixed at issue makes its raw output gradient −weight delta. In parameter space this is multiplied by the native action Jacobian; it is a stochastic supervised improvement direction, not a guarantee of deployment gain. Parameter clipping/Adam, target generalization and finite-radius bias remain part of the experiment. Known randomized search and Taylor arguments are the primitives; the integrated native learner/consequence receipt and charged causal implementation are the construction under test.

## Economical receipt boundary

Carry one current activation until two learning decisions have matured; retain the exact learner snapshot only for admitted counterfactual work. The factual future trajectory already supplies baseline outcomes. One alternative world avoids a dense four-direction or full parameter derivative teacher, but performs real subsequent own learning, native decisions and refresh work. Charge that work before asserting economy. Admission and residual correction from AWS can later bound receipt work; this experiment first checks that the longer-horizon target is useful.
