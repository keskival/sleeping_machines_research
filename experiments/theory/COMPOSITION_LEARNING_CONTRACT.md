# A learning contract for compositions of efficient event learners

16 October 2026. R1 integration theory; motivated by the founder's question: if the stages are efficient learners, what makes their composition learn?

## 1. The object being composed

At stage k, events E_k schedule work, accessible persistent memories M_<=k carry content, and a transition emits events and versioned refinements:

    U_k : (E_k, accessible M, own numerical state) -> (E_(k+1), delta M_k).

The interface includes identities, roles, timestamps and queryable lower versions. An emitted winner ID is a routing observation, not automatically an abstraction. A stage can consume many events and refine a memory that is read many times before another update. Representation depth, event rate, query/decision count, state lifetime and credit horizon are separate axes.

An efficient stage learner provides evidence for this construction, not a theorem that any interconnection learns. Its input distribution and objective change inside a learned hierarchy.

## 2. Information: what must remain accessible

Let H be the available causal input history, Z the upper stage's interface and Y the task target. Assume Z is generated from H without additional information about Y: Y -> H -> Z. The optimal log-loss prediction gap is

    inf_q E[-log q(Y|Z)] - inf_p E[-log p(Y|H)]
      = H(Y|Z)-H(Y|H) = I(Y;H|Z).

This is an exact statement about optimal predictors, not a claim that mutual information is currently measured or optimized. It states the obstruction: no upper learner can recover task evidence excluded from Z. Queryable lower memories can expand Z on demand; persistent access is not permission to use unrevealed future observations. Stage-specific likelihoods need not preserve all information required by the final task.

## 3. Expressivity: joint content and persistent refinement

Use a shared addressed memory space with compatible transport:

    M_(k+1) = M_k + alpha_k g_k(M_k, ReadRace_k(M_k), E_k).

A write can depend nonlinearly on multiple participants with stable roles/identities and timing. Adding independently classified children is insufficient for general relational tasks; nonlinear joint terms must be available before the final decision. Existing identity/role composers provide such terms. Additive storage does not imply additive reasoning.

For shared linear temporal transport D, D(sum_k M_k)=sum_k D(M_k). Contributions can evolve and be updated at different times. A negative refinement can cancel lower content numerically; retaining separate versions/contributions preserves access, not immunity to cancellation.

Setting a new contribution's gate to zero nests the smaller predictive family inside the larger one, provided other parameters and candidate semantics are preserved. This proves attainable optimum containment, not monotonic training or generalization. With delta M=alpha*g_theta and alpha=0, theta initially receives zero task gradient; a nonzero initialized branch can teach alpha. Setting both factors to zero can prevent this opening signal.

## 4. Credit: the assembly and its producers must be taught

For a sampled assembly A~P_theta(A|causal history),

    grad E[ell] = E[grad ell |_A + (ell-b) grad log P_theta(A)],

where the baseline b must be independent of A conditional on the history (or be accompanied by the appropriate correction). The first term teaches selected content and deterministic computations. The second teaches discrete assembly decisions. Without-replacement reads require the actual joint law over legal distinct participant tuples; independent role alternatives can teach an impossible duplicated assembly.

Persistent content makes producer credit essential. If a stored write W_theta(s) is useful at later time t, the task cotangent must reach the relevant write and its content sources. For a finite, fixed recorded producer DAG, reverse accumulation over its local Jacobians computes its exact conditional VJP. Native own-write eligibility can supply the numerical receiver-state derivative without retaining the event graph; local input/write replay supplies complementary producer-input derivatives, with frozen write coefficients avoiding duplicate own-parameter credit.

This does not prove whole-history exactness. Fixing historical route/timing decisions, cutting base-input/inbox provenance, truncating child replay, or replaying under changed parameters omits or changes paths. Each approximation belongs in the declared credit contract. Full sequence-gradient claims require accounting for all such paths.

For literal timed point-process emissions, event credit also includes silence: log lambda_c(t) minus the integrated total hazard. Token-likelihood races alone do not establish that this timed-emission objective has been integrated.

## 5. Why a bounded contract can still be useful

Suppose an L-smooth bounded-below objective is optimized with a gradient estimate whose conditional bias has norm <=epsilon and conditional variance <=sigma^2. Under a sufficiently small constant step size eta, standard descent plus Young's inequality gives an average squared-gradient bound of order

    Delta/(eta*T) + L*eta*sigma^2 + epsilon^2.

This explains the engineering targets: finite signal, controlled variance and sufficiently small omitted-path bias. It is not a convergence guarantee for our current learner: those assumptions, especially a uniform bound on its changing-parameter/truncated-credit bias, have not been established. Nonconvex stationary-point progress also does not imply a global optimum or scaling supremacy.

Residual addition contributes a direct derivative to each memory contribution, plus indirect paths through later readers. It does not bound the readout Jacobian or prevent gating/normalization saturation. We must inspect the actual task signal rather than infer trainability from an identity connection.

## 6. Work and causality

A version-bound child graph descends strictly in abstraction level. With branching b and at most h replay edges along a path, unrestricted node visits are bounded by a geometric sum through h. An explicit global edge budget B gives <=B additional child reads per root, independent of the age of a stored memory. Lower budgets trade credit coverage for learning work; they do not reset numerical state.

Charge key discovery, selected values, losing alternatives, pair interactions, temporal transport, receiver traces, producer replay, optimizer work, lineage storage and traffic. A bounded replay budget does not bound the transitive lineage store or all candidate discovery. Soft firing is a deterministic training relaxation; it neither guarantees an unbiased hard-firing objective nor demonstrates sparse dispatch.

All queries read versions available at their issue times. Forecasts precede target-driven updates. Persistent numerical state, credit storage and gradient truncation remain separate.

## 7. Concrete integration

`sleeping_machines/compositional_producer_credit.py` extends the legal joint-query native core with actual-child delayed content replay. No parameters or factual RNG draws are added. New publications save detached child IDs, issue/reply times and assembly-ready time. Historical producer replay reconstructs the original role alignment and injects zero-forward child VJPs into the saved participant views. Default two-edge path depth and six-edge total budget bound each root's extra replay. Old publications retain their original state without fabricated lineage.

This repairs one precise missing path: a later upper read can now teach lower composers used by its historical producer. It retains native temporal transport, sparse selected writes, indexed episodic memory, identity/role joint assembly, legal joint query credit and receiver eligibility. It does not add historical query-selection credit, recursive factual base-input/inbox provenance, asynchronous optimizer updates or exact changing-parameter history derivatives.

Required gate: bitwise factual predictions/main and alternative RNG parity; detached immutable lineage; a delayed upper loss produces nonzero lower-composer credit absent in one-hop replay; its composer/mix VJP matches an independently rebuilt finite conditional DAG; explicit work cap and serialized continuation; then a private acquired eight-layer GPT-2/FineWeb optimizer/recovery gate. This is integration, not a predictive win. Active source-bound fits finish unchanged. No further acquisition is admitted from this note alone; existing race-memory scaling orders remain first.

## 8. The competitive decision

The decisive evidence is deeper compositions improving same-protocol held-out prediction and reusable relations at acceptable full learning/execution work. Diagnostic derivative correctness removes a concrete obstacle; only completed learning evidence establishes the competitive gain. The strongest path joins the demonstrated causal race law, information-preserving persistent memory, joint refinement and bounded task-aligned credit. The contract makes those obligations explicit and testable.
