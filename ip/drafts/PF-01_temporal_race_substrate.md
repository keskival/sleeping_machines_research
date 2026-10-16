# PF-01 — Temporal race processing with sparse addressed state and alternative-route learning

CONFIDENTIAL — inventor working draft, 15 October 2026. Inventor: **Tero Keski-Valkama**, according to his express statement that he made the inventions with AI-agent support. Applicant/assignee: to be determined from executed ownership documents. No filing, priority claim or declaration is made here. Example claims require prior-art, disclosure, support and jurisdiction review. Figure descriptions are instructions for drawings, not finished drawings. Implementation status is stated separately from proposed embodiments.

## Disclosure route

Earlier-public US grace-period candidate. IDF-06a–f dates and the older manifesto must be checked feature by feature. Do not present the same public foundations as novel EP matter.

## Abstract

A processor maintains addressed state units, separate selection and payload paths, and scheduled deliveries. Candidate clocks determine addressed activity and computational time. Selected payloads update persistent local state; a learning processor evaluates consequences of specified unselected routes without executing those routes as factual deliveries. Absence intervals can contribute supervision or trigger scheduled queries. The disclosed system separates allocated state, scored candidates and committed writes, permitting memory capacity to exceed activity while making candidate discovery and teaching work explicit.

## Technical field, background and detailed description

The field is event-driven computing for technical monitoring and adaptive processors. Dense updates can require reading or writing state that does not participate in the current observation. A hard sparse route reduces deliveries but can prevent ordinary path differentiation from teaching the choices that were not made.

An input packet contains a source, destination or candidate-address set, payload, observation cutoff and modeled readiness time. Unit i stores a memory vector and last-update timestamp. On access at time t, a representative mode transports its prior state as z_i(t)=exp((-r_i+i*omega_i)*(t-t_i))*z_i(t_i). A local map combines transported state with incoming content. A key path determines a candidate clock or score; a separate value path determines the delivered content. Only selected deliveries commit a factual state write.

For one embodiment, lambda_i=exp(s_i), E_i are independent unit exponential variates, and the selected candidate minimizes E_i/lambda_i. This samples the categorical probabilities lambda_i/sum(lambda). This identity is established mathematics; the candidate claim concerns the concrete state, message, schedule and teaching arrangement. Another embodiment delivers a declared set of value packets with delay-coded weights and aggregates all required arrivals. Exact full-support attention aggregation is not asserted for a winner-only or truncated delivery.

A training receipt identifies the factual state and an alternative action. The processor can replay the changed write under coupled future observations to obtain a loss contrast. Probability-weighted or sampled residual credit is delivered to the route parameters. A losing unit's state is not changed in the factual world merely because it was evaluated for learning. A no-event interval may contribute a survival integral to the loss or cause a deadline event whose readout includes elapsed silence.

Software timers, event queues and delay-line circuits are alternatives. Modeled computation time, observation availability and physical runtime are distinct. Ties, finite precision, simultaneous arrivals, cancellation and a finite stopping/resource rule must be declared. Additional capacity can increase scoring and training work even at fixed selected writes.

## Example claims

1. A computer-implemented method for processing a technical event stream, comprising maintaining timestamped addressed memory units; computing candidate selection clocks through selection paths distinct from payload paths; selecting a bounded subset of candidate deliveries using the clocks; transporting and updating memory at selected addresses using the delivered payloads; and training selection parameters using a receipt evaluating a specified unselected delivery while retaining the factual committed state separately.

2. The method of claim 1, wherein candidate clocks comprise exponential timers and a first timer selects a committed delivery.

3. The method of claim 1, wherein allocated receiver count exceeds the number of selected memory writes per input event and key-discovery work is performed separately from value delivery.

4. The method of claim 1, wherein an alternative action is evaluated from an issued state with coupled exogenous observations and recorded random draws.

5. The method of claim 1, wherein an interval without an observed event contributes survival supervision or a scheduled query.

6. The method of claim 1, wherein timestamped lazy transport replaces periodic updates of unaddressed memory units.

7. A processing system comprising addressed state storage, a timer-selection circuit or scheduler, a payload dispatch interface and a learning processor configured to perform the method of claim 1.

8. A non-transitory computer-readable storage medium storing instructions that cause a processor to perform the method of claim 1.

## Drawings, support and review notes

Figures: (1) separate key and value paths around a timer race; (2) factual and alternative state worlds; (3) dormant and selected memory regions. Sources: IDF-06; theory notes 08, 22, 34, 59; model_family_specification.md. Exact softmax races, attention, sparse experts, survival training and neuromorphic timing are prior-art search targets. Existing public descriptions may defeat broad claims even in the US if external or older art applies. No measured physical energy or latency advantage is asserted.
