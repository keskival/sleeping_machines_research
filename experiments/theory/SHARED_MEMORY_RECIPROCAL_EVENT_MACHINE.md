# Shared memory, independent computation, reciprocal teaching

Curie, 12 October 2026. User-directed B1/R1 shared enabler and B5 causal-development integration. This formalization extends [reciprocal event learning](RECIPROCAL_EVENT_LEARNING.md) and [the learner-inclusive world model](SHARED_WORLD_MODEL_AND_RECIPROCAL_CREDIT.md). It proposes an implementation and derives conditional identities; it reports no new benchmark win.

## Architectural decision and its concrete motivation

Use the native temporal/race family for both observation processing and credit processing. Each has independent learned discovery, clocks, reads, writes, retention and messages. They share addressed memory, rather than requiring the critic to replay the forward access sequence. Standard local projections/readouts support these operations; a dense recurrent controller is not the credit substrate.

Current RaceCreditSensor uses independent native credit units but stores node state and relationship-credit state separately. The critic reads detached forward features; credit-only evaluation does not let critic state affect forward forecasts. The joint arm subtracts a cached scalar from source scores, rather than offering independent shared-bank access. These are concrete missing reciprocal paths, not limits of the model family.

Completed private causal evidence: raw native credit beats zero-credit and reset-memory controls in all three confirmation seeds, exact route teaching in two. Strict transfer gate fails. Centering its teacher lowers audited contrast error/magnitude from91.7 to1.154 while worsening selection NLL0.613521 to0.621341. Therefore unweighted critic regression accuracy cannot be the adoption objective. Shared-memory operations must be taught and evaluated through actual future task outcomes.

Retain temporal evolution, hard races, separate keys/values, persistent deep representations, sparse selected writes and paid counterfactual credit. Add independent access to common records and causal teaching of its consequences. Neither shared storage nor typed messages imposes identical actor/critic parameters or access decisions.

## One causal event machine

Let H_t contain only information available before a decision. The augmented state is

    X_t = (M_t, theta_t, phi_t, optimizer_t, issued_t, versions_t, clocks_t).

M is a sparse bank of records (address, key, value, arrival, version, provenance). An event has role r in {F,C}, available content e, and computational arrival time tau. Native parameters eta_F=theta and eta_C=phi implement

    a ~ pi_r(. | e, X),
    (message, writes, delay) = T_r(e, X, a),
    X' = Apply(X, writes, message, delay).

An action includes discovery, read selection, write destination and update proposal. Discovery and losing proposals cost work. Writes become visible at completion, not issue. A read binds the visible record version. Concurrent completion order is part of the transition law; independent addresses may commute, overlapping writes generally do not. Logical event/version records enable reproducible execution without requiring a global computational barrier.

Role is message metadata, not a required separate bank. Initially preserve observed-state records and allocate bounded credit writes as additional typed records in the same address space. Either role may read them. This avoids making untrained destructive overwrite the first test of sharing. Later learned overwrite/merge is an explicit action with its own utility and counterfactuals.

Parameters can overlap, but shared parameters receive the sum of the declared objective paths, not two unaccounted independent optimizer steps. Initial integration keeps theta and phi distinct while sharing numerical records.

## What reciprocal teaching means mathematically

Use a finite causal horizon, with task losses scored before their labels arrive:

    J = E[sum_t w_t ell_t + lambda W_t + mu D_t].

W charges complete work; D charges latency under a declared convention. One common task/resource objective evaluates both roles. Auxiliary loss prediction or calibration teaches useful representations but does not replace J.

For an issued operation a, define Q_r(H,a)=E[future cost | H, do(a)] under a specified continuation learning policy. A local operation is taught by its finite intervention effect

    Delta_r(a,b) = Q_r(H,a)-Q_r(H,b).

The continuation includes future theta/phi updates and memory evolution. Critic writes therefore have both a memory pathway and a learning pathway. If g_phi is the critic's update proposal,

    theta' = U(theta, optimizer, g_phi),
    d J / d phi = J_M d M'/d phi
                  + J_theta' U_g d g_phi/d phi
                  + J_optimizer' d optimizer'/d phi
                  + other declared continuation-state terms.

This is a one-transition chain-rule decomposition at fixed incoming state, not an exact derivative through the complete previous learning history. Smooth terms coexist with score-function/forced-action terms for hard routes. Omitting the shared-memory term would miss precisely the new architectural capability.

For a sampled hard action, conditional on H, the score identity is

    grad_eta E[G | H] = E[(G-b(H)) grad_eta log pi_eta(a|H)]
                        + E[pathwise derivatives of G].

Here b is independent of the sampled action given H; continuation terms cannot be silently dropped. Native counterfactual surrogates may approximate this target differently: they must retain their declared target and bias rather than be called exact merely because the units are native.

The later forward computation supplies evidence about a critic operation. A native credit event attributes that evidence to its eligible producers and alternatives. Likewise, critic outcomes teach forward producers. This is reciprocal teaching, not literal reversal of wall-clock time. No third learned teacher is mathematically necessary: delivered task losses ground the loop; sampled interventions and exact local algebra can provide independent anchors.

## Learning without an infinite stack of critics

Use the same reusable operation-effect function C_phi(H,a,b,horizon) to estimate consequences for either role. Operation tokens include role, producer, consumer, memory address/version, local activation, elapsed time, optimizer context and horizon. An address retrieves state; it need not be a trainable identity token. Shared relational features permit transfer to unseen addresses.

An operation emits a compact eligibility record. Later loss events retrieve a bounded set of such records through native learned races. They can teach phi's own earlier memory choices as well as theta's choices. Parameter updates happen after the relevant issued prediction is scored; weights used to construct a teaching estimate are held fixed for that estimate.

An independent audit A with predictable positive probability p gives

    corrected_Delta = predicted_Delta
                      + A/p (observed_Delta-predicted_Delta).

Conditional expectation equals the audited intervention target when the audit observes an unbiased sample of that target. Auditing a local decoder difference does not observe a full future continuation effect. Real shared futures require paid branched replay on recorded exogenous inputs or randomized intervention evidence; on action-dependent environments a replay is not automatically valid.

The anchor prevents prediction error from silently redefining the chosen teaching target. It does not guarantee low variance, optimizer-unbiased updates after clipping, convergence, or identification from insufficient observations. Self-teaching alone can sustain a mutually agreeable but wrong prediction/credit loop; delivered outcomes and supported interventions keep it grounded.

## Why shared memory could be economical

Forward and credit events can reuse records containing observations, timing, learned relations and eligibility. The critic need not reconstruct a second world representation or duplicate an entire forward trajectory. Conversely a useful critic write can be served to many subsequent forward events.

For N available records, K discovered candidates, k selected accesses and B retained eligibility records, an intended event cost is

    discovery(N) + O(K d_key + k d_value + B d_trace) + update_work.

This is a target accounting structure, not measured sublinear discovery. Exact top-k discovery can still scan N. At fixed k, growing capacity does not imply constant learning work: proposal scoring, audit coverage, contention, metadata and optimizer touches all count. Native parameter sharing can avoid per-address learned weights, but useful memory grows with retained records.

Expiry/retention is itself a learned write decision: retaining every critic activation defeats the economy. Teach survival of records by future utility per retention/access cost, with bounded traces and delayed feedback. This trades exact full-history differentiation for a specified learned estimator and audit protocol.

## Stability, depth and what intuition suggests

Separate three axes: depth of inference, causal event horizon, and depth of the teaching dependency graph. A critic that takes a short route to a deep producer could bypass repeated inaccurate layerwise transport. It cannot infer an unidentifiable consequence without observations or useful structure.

For a fixed local linearization, write stacked memory perturbations as

    [delta M_F'; delta M_C'] = [[A,B],[C,D]] [delta M_F;delta M_C].

A sufficient norm condition for contraction is the spectral radius of the nonnegative block-norm matrix [[||A||,||B||],[||C||,||D||]] below1. In particular a<1,d<1 and bc<(1-a)(1-d) suffice for that2x2 bound. It includes the new bidirectional feedback gain. Individual decay a,d<1 alone is insufficient. This is a fixed local transition condition, not global stability of changing weights or arbitrary hard races.

Practical intuition: start with bounded residual critic writes, causal visibility and independent clocks, then learn when additional coupling pays. Persistent memory should preserve information; contraction need not be imposed on every state dimension or erase long-term evidence. Exact copying/invariants and controlled transient feedback can coexist. Stability can also be measured over event blocks rather than demanding per-event decay.

Shared structure can reduce compounding approximation errors by learning a direct relation between a future loss event and a relevant deep operation. The decisive evidence is an improving full-work ratio across neural depth, not just larger generating-graph depth or better scalar critic fit.

## Concrete owned integration test

This is a new mechanism test inside B5/R1, not the deferred broad transfer of the current failing-gate variant. Before fitting, clone immutable completed drivers; preserve the existing evidence.

1. Introduce a sparse common record store with versioned causal reads and queued writes. Forward and credit roles independently select reads/writes with native race units. Observational records remain available; additional bounded residual records use learned relational keys.
2. Make the forward read actually consume selected credit-written content, not merely a cached scalar source penalty. Make the critic read independently selected shared content, not only detached replay features. Save compact operation tokens for both roles.
3. Teach critic write/read choices by subsequent supervised-loss effects. A first paid diagnostic branches a bounded continuation from the same issued state with and without one critic write, using identical future exogenous events, actor/critic RNG and initial optimizer state. Charge both branches. Branch consequences include future learning. Never consult hidden generator metadata.
4. Compare separate memory, shared critic-read-only, shared reads+writes without future-utility teaching, and fully reciprocal sharing. Match native parameter capacity, event order and update opportunities; charge bank size, discovery, proposals, audits, all optimizer work and inference accesses. Freeze/reset controls isolate useful acquisition from extra memory capacity.
5. Numerical contracts: no prediction sees later feedback; independent access choices; exact replay of bound versions; real future-loss change through a critic write; finite-difference parity on a small smooth local witness; shared-bank forward/critic parameter changes. Numerical correctness is not the fit endpoint.
6. After resource smoke, one bounded development fit. Evaluate prequential task loss and complete CPU work, plus measured critic-write intervention utility. Require useful coupling before selecting a neural-depth4/8 comparison and fixed-seed confirmation. No automatic large-data transfer or publicTEST.

The central hypothesis is a learned causal memory system whose computation and learning share representations, while each independently acquires the information it needs. Its potential advantage is reusable, task-valued credit structure and sparse teaching across time and depth. Shared banks are the enabling path; demonstrated future learning per complete work is the win criterion.

## Implemented reciprocal path and diagnosed teaching split

SharedBankSensor v2 now implements independent native hard-race reads and receiver writes, with both roles reading common observation/credit records. Later factual forward losses give nonzero gradients to earlier native critic write parameters and its read policy. Completion-time filtering and a matched untaught control pass; the control retains actor-reader learning. Bank destinations remain public relational addresses, visibility binds numeric record versions, and gradients cover eight-feedback windows. This is the memory pathway above, not a complete optimizer-aware historical derivative.

Completed first four-arm selection NLL: separate0.615585, critic-read-only0.617031, shared-untaught0.616819, reciprocal0.616480. Reciprocal improves untaught but does not beat separate;346.113 versus237.261s fitting. Retain the raw native-credit incumbent0.613521/159.354s. The result tests this first shared-bank construction, not the family.

Paid future-update diagnostics on twoTRAIN prefixes find auxiliary total gradient norms600–1200times task-memory norms, slight negative component cosines and tiny next-loss effects. Norms include the utility readout, so they do not certify per-coordinate optimizer dominance. Regression updates worsen next16forecast loss; task updates barely change it relative to momentum coast. Source-bound evidence is curie_shared_bank_20261012_v1_future_diagnosis.

The diagnosed construction splits phi=(psi,omega): native memory/access parameters psi receive factual task teaching through shared records; the same linear utility readout omega receives revealed-loss regression on detached z_psi. This retains one native representation rather than adding a separate dense controller:

    memory teaching: grad_psi J_task through subsequent shared-memory use,
    readout teaching: grad_omega (c_omega(stop_gradient(z_psi))-revealed_loss)^2.

Separate clipping budgets prevent the readout norm directly rescaling actor/memory gradients. This block teaching rule is not asserted to be gradient descent on one global potential, and its future-loss/credit-utility performance must be measured. The task-taught fit is an owned diagnosed engineering comparison, not a new mathematical convergence claim.

An inference gain from additional critic memory and a gain from improved learning credit are distinct endpoints. Fixed-weight serving with acquired memory, memory-reset/read-disabled controls and unchanged-memory credit-output controls should separate them in confirmation. Shared representation is intended to support both; evidence for one pathway cannot silently establish the other.

## Public address semantics repair a content-alias obstruction

Task-taught sharing completes at reciprocal0.616242 versus untaught0.615281; no adoption. A separate representational gap is now identified: the global bank read law is pi(a | q(z_node), visible_content), without the node address relative to record endpoints. If two distinct nodes have identical z_node and the same visible bank, their read laws are identical. When the useful stored evidence differs by node, that policy class cannot retrieve different evidence for those two queries. Enlarging its content-only projection does not remove the missing input.

Public node/relationship equality repairs the gap without a learned identity table. AddressedSharedBank discovers records indexed by the reader's known endpoints, then global fallback. Its learned relative score adds role and target/source-equality features. Under a bijective public address relabeling, equality, numerical state correspondence and insertion order are preserved; the entire read/write policy is equivariant under matched randomness. These syntactic operation semantics are prescribed. Which relationship/role is useful remains learned; no hidden generating parents or gates enter the model.

Active-readout relabel parity, future-write invisibility, retrieval outside global recency and nonzero task teaching of relative critic key weights pass the integrated contract. The bounded matched addressed fit is in development. This is the next specific native construction, not evidence of acquired unknown graph structure or a scaling win.

## Preserve outcome information in credit messages

The addressed fit completes at reciprocal0.615836 versus untaught0.615548. Correct access semantics alone does not earn adoption. Inspection identifies another information bottleneck: delivered credit tokens encode(kind,loss,auditflag), with the flag duplicating kind. Binary loss alone is not an outcome-preserving message. At p=1/2, either target produces BCE=log2; target information is completely erased by that scalar.

OutcomeAddressedBank replaces the redundant flag with the actual revealed target, using the same payload dimension and parameters. For0<p<1, the pair(loss,y) reconstructs p as exp(-loss) wheny=1 or1-exp(-loss) wheny=0. It retains the outcome and signed-error information without requiring the critic to infer the actor's changing output from upstream features alone. Requests contain no target; delivery/completion still gates shared visibility. The contract exhibits different native credit states for opposite labels at exactly the same0.5forecast/loss, with unchanged issued forecast and parameter schema. Numerical state difference0.220187 is a channel witness, not a prediction-quality gain.

The bounded matched fit tests this final diagnosed construction in the current integration round. A next-stage credit-output learning target must distinguish factual immediate-loss prediction from the actual future learning value of a route/update. The memory path implemented here does not differentiate through optimizer history. Delayed paid continuation contrasts, with source states/versions and actual moment updates bound, are the appropriate target for that separate pathway; retain source-matched controls and charge replay. Additional memory serving capacity alone is not evidence of cheaper deep learning.
