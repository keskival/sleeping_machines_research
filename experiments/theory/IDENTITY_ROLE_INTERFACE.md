Current continuation is governed by [data and optimization scaling](COMPOSITION_DATA_AND_OPTIMIZATION_SCALING.md):the64update result does not justify a capacity verdict,and the user has authorized an adequate fixed-architecture test before further redesign.

# Compositional continuation after native identity acquisition

16 October 2026. No new ablation campaign or automatic fit.

The completed integrated actor raises observable part-shape DEV accuracy from
12.5% to50.1953% (257/512parts). Whole-label DEV NLL improves against its own
corrected-ownership initial actor,0.707531 to0.692212,but does not beat the
saved acquired actor0.686609. TRAIN shape accuracy51.5625% is also incomplete.
This identifies useful local learning and an unfinished composition problem;
it does not identify one proven cause of the remaining class error.

For a scene containing the same multiset of identities in every class,any
permutation-invariant function of identity counts is constant across classes.
The sufficient interface must retain joint identity-and-role information,
not merely separate identity and position marginals. A useful local object is
B = sum_i phi(z_i) tensor psi(r_i),with additional ordered pair terms
sum_{i != j} a(z_i,z_j,r_i-r_j). Known tensor/bilinear primitives supply these
interactions;our contribution must be their trainable native temporal,
persistent and sparse composition. An expressive formula alone proves no
learned scaling advantage.

The current actor already includes bilinear child-role interactions. Adding
another bilinear layer without checking its information interface repeats the
same hypothesis. In `RelationalEventCore.consume_event`,the cache is keyed by
(group,quadrant):only the latest view in each slot reaches that explicit
relation operation. Native event state still consumes every accepted message,
and versioned directories retain more complete child references. Therefore
neither 'all earlier information is lost' nor 'all children are jointly bound'
follows from the code. Multiple identities may share one geometric role.

The next architectural construction should use the existing versioned part
directory as the addressable set of participants. Keep identity/address
separate from geometric role;coarse role must not be a unique storage key.
Read the exact child version at the causal time it was issued/used by the
parent. The current shape teacher instead reads latest level-one state at the
drained scene endpoint,so its50.2% score is not a measurement of every emitted
child view. Aligning local teaching with actual parent-consumed versions makes
the objective act on the information interface that higher levels receive.
This is an engineering priority,not an attribution established by an ablation.

Preserve native time evolution,competing clocks,sparse state writes,
counterfactual credit and generated views. Use race selection over the
addressable participant set;charge candidate discovery,all scored keys,
selected values,pair work and learning. Bound pair interaction work by selected
activity,while retaining unselected participants in persistent capacity.
Do not turn the directory into a dense replay of all events.

Before another acquisition:implement identity-preserving role storage and
version-aligned local targets in one integrated candidate;validate causal
availability,no key collision,no role permutation aliasing,finite deep credit
and resumable state. These are correctness contracts,not ablation arms.
Reuse the completed actor,data and TRAIN-fitted vocabulary. Select on whole
DEV loss while reporting local identity separately;keep TEST closed. No
unchanged longer identity fit is admitted by this note. Existing16M language
depth acquisitions finish unchanged and answer the separate depth/data scaling
decision.

Evidence: `experiments/results/hierarchy/aws_g_hier_part_identity_20261016_v1.json`
and `investment/part_identity_development_evidence.json`.

## Integrated implementation and admission

`identity_role_composition.py` now keys persistent participant views by child
level/address,with memory version and pose retained separately. Same-role
children are preserved and refreshed by identity. Symmetric pair products and
antisymmetric content products multiplied by signed relative pose make pair
aggregation independent of enumeration while retaining content/pose binding.
The original native receipt and timer races,protected temporal modes and
counterfactual receipt learning remain active. All current participants and
quadratic pair work are charged;this is not sparse partner discovery.

Local shape teaching reads immutable level-one versions actually retained by
parent publications,at their issue times. Its endpoint differs from the old
latest-memory classifier;their accuracies are not a matched before/after
comparison. The acquired learned-identity actor is the starting checkpoint.
Whole-label DEV and the saved0.686609 reference remain the selection criterion.

Twenty-two combined numerical contracts pass,including same-role identity
retention,content/pose sensitivity,pair enumeration invariance,deep gradients,
causal version reads and exact no-grad in-flight checkpoint recovery. Recovery
rekeys Python envelope identities through stable queue serials. It preserves
numerical inference state,not an in-progress training credit graph.

The actual paired optimizer gate passes in6.114seconds at804980KiB peak RSS,
with finite nonzero content gradients at all eight levels. One64paired-update
integrated acquisition is admitted after that gate,with the same fixed
whole-label CE+0.25local identity objective,source-bound8update checkpoints,
TRAIN-only readout and untouched TEST. No ablation/control fit or automatic
longer continuation is admitted. These are implementation/resource results;
the completed prediction endpoint decides whether to promote this candidate.
