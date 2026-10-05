# Size the model from measured token scaling

User direction,5October: once the general setup learns well and has passed
fault checks, obtain rough scaling laws to choose model size for training data.
This follows the small-fit mechanism repair stage; it does not admit a grid now.

Freeze the winning tokenization, input/readout initialization, routing/learning
rule, credit horizon, cadence, optimizer and development scoring protocol.
Check that fits learn beyond the train-frequency baseline and that larger
capacity lowers approximation error when sufficiently exposed. Fix accounting,
streaming data allocation and exact continuation before long scaling fits.

## Quantities kept separate

For every completed cell record vocabulary/lexical parameters, shared core
parameters, private receiver parameters, total learned parameters, available
receiver states, selected writes/value deliveries, scored keys and learning
proposals; persistent-state bytes; unique training-data tokens, actual fitting
target presentations and passes; complete fitting work/per-target work and
inference work, measured CPU wall and peakRSS. Initialization frequency counts
also consume training data and are included in the declared data population.

The first width sweep holds depth/heads/pool/credit fixed, so it is one model
family. A separate pool sweep tests capacity beyond activity. Do not collapse
those axes into one dense parameter count or apply a borrowed tokens-per-
parameter constant. Changes to optimizer/credit/protocol start a new sweep.

## Economical first measurement packet

Choose three widths around the successfully debugged integrated fit, and three
training-data budgets (initially64K/256K/1M tokens if the learning curves support
those scales). Use the same fixed number of data passes and a fixed number of
validation checkpoints per pass, with8 evaluation lanes and one sufficiently
large development interval disjoint from public benchmark validation.

Admit the smallest three cells first. Advance promising widths to the next data
budget, retaining enough crossed cells to identify model/data effects. Aim for
nine completed cells for the joint rough fit; no all-knob search. Repeat one
central cell to estimate variation. Stop a cell only for a predefined numerical,
resource or learning failure, retaining its record. Use frozen small-run
checkpoints where continued data/protocol remain comparable.

Fit an exploratory loss surface

    L(N,D) = L_floor + a*(N/N0)^(-alpha) + b*(D/D0)^(-beta),

where N is learned parameter capacity along this fixed-width family and D is
unique data at the fixed pass count. Report uncertainty and residuals; fit a
simpler separate-slope model if the joint parameters are poorly identified.
An error-balanced size relation is N(D) proportional to D^(beta/alpha), with its
constant fitted from these cells. It is not a universal compute-optimal rule.

Use measured c(N)=complete fitting work per target, including readout,
counterfactual probes, cache refresh and optimizer. Given a CPU/work budget,
scan the measured/interpolated size choices with D_presentations*c(N) within
that budget, and choose predicted validation quality. Available capacity and
active cost are separate for this substrate, so dense C proportional to N*D is
not assumed.

Reserve one next-larger data/size point before fitting the law, e.g.4M tokens
if1M is the last fitted scale. Predict its loss and cost, then run it. Its
prediction error determines whether the rule is trustworthy enough to select
the public-benchmark model. A failed extrapolation triggers diagnosis/new fit,
not an unexamined hundred-million-token investment.

No scaling coefficients have been measured for the new tokenized setup yet.
Current8K pilot variants are mechanism comparisons, not a scaling dataset.

## FLOP frontier and sparse capacity: 5 October clarification

The user targets lower language loss at equal complete fitting FLOPs and useful
capacity growing faster than selected activity. Treat this as the selection
criterion, not a constraint imposed on fitted coefficients. Lower loss at a
given compute budget and a steeper decline in reducible loss are distinct
claims; reserve a larger point to test extrapolation and possible crossover.
Dense Chinchilla allocation does not prescribe total sparse parameter count
or persistent state size. Nor does additional stored state prove additional
useful representational capacity. Measure quality and utilization as it grows.

Dense reference: Hoffmann et al., Training Compute-Optimal Large Language Models
(https://arxiv.org/abs/2203.15556), empirically balances model size and training
tokens. Routed reference: Clark et al., Unified Scaling Laws for Routed Language
Models (https://proceedings.mlr.press/v162/clark22a/clark22a.pdf), fits log loss
in active model size and effective expert count with an interaction and expert
saturation. Its fixed 130B-token training regime is not a compute-optimal
data-allocation sweep. Krajewski et al., Scaling Laws for Fine-Grained Mixture
of Experts (https://arxiv.org/abs/2402.07871), varies data and expert granularity
and finds an increasing modelled compute advantage for optimized MoE. Its
large-budget extrapolations are predictions, not completed fits at those scales.
Thus sparse Transformers are substantive scaling competitors; beating a dense
reference alone does not imply beating the best sparse Transformer frontier.

For our family separately fit available capacity, selected work, discovery,
credit work, data exposure and persistent state. Keep lexical/readout work
explicit. Increasing pool size currently increases scored keys and may increase
optimizer work even when selected writes stay fixed. Scalable discovery and
learning are implementation objectives, not assumed zero-cost properties.
The current 2K/8K pilots differ in passes and evaluation populations and do not
identify a scaling exponent. The admitted 64K test holds the 8K development
population and two-pass protocol; width/data crossed cells follow its learning
gate. No empirical exponent or below-Transformer law is claimed yet.

## Report visualization — user request, 5 October

When sufficient crossed data/capacity evidence supports the first estimation,
visualize the measured cells and fitted laws prominently in the report. Use
standalone reproducible plots suitable for export, with source/result provenance.
Show held-out NLL against complete fitting FLOPs; capacity/data panels distinguish
available parameters/state from selected activity and scored/learning work.
Mark measured points, exploratory fitted curves, uncertainty and the reserved
larger-point prediction versus its observed result. Keep special-function and
compute-boundary conventions explicit. Display raw measured points before any
fit is identifiable; do not manufacture a scaling curve from one data budget
or fill pending points with predictions presented as measurements.

The planned crossed64K/256K/1Mpacket and central seed repeat provide the first
rough surface; if coefficients are poorly identified, plot the supported
simpler slopes and uncertainty instead. The4Mpoint remains an extrapolation
test. Report the modelled estimate as an estimate, and retain every valid raw
point, including losses. This visualization is an explicit deliverable.
