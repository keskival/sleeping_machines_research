# One event interface, several reception programs

This is a **worked semantic example**, not a trained model or benchmark. It
shows how local policy changes the function a composition computes. The
[interactive atlas](architecture_atlas.html) lets you change the arrival of C,
the reception interval and the causal query cutoff.

Four observed messages enter the same receiver:

| Message | Arrival time | Scalar content |
| --- | --- | --- |
| A | 0.0 | 1 |
| B | 0.8 | −2 |
| C | 1.6 | 3 |
| D | 4.0 | 4 |

Take interval H=1 and query cutoff q=5. The local state is the sum of accepted
content, with no damping. Later messages can start a subsequent group; the
example reports the **first group only**. Arrivals at a timer's deadline are
processed before that timer. All policies use only the observed prefix t≤q.

| Receiver policy | First group's members | First emitted content | Emission time | Computational / representational distinction |
| --- | --- | --- | --- | --- |
| First arrival | A | 1 | 0.0 | One message determines the continuation |
| Window anchored to first arrival | A, B | −1 | 1.0 | Integrates evidence in a fixed interval after A |
| Silence timeout / popcorn | A, B, C | 2 | 2.6 | Each admitted arrival extends the deadline; the group survives while gaps are ≤H |
| All arrivals through a fixed query cutoff | A, B, C, D | 6 | 5.0 | Full support over the declared prefix; the external query closes reception |

The same messages produce different functions because **reception is an
operator**. More deliveries can preserve complementary evidence; they also
cost integration, traffic and time. A deadline supplies information about
absence. It does not require a network-wide periodic update.

If q=2, the popcorn receiver has accepted A, B and C but has not emitted: its
deadline is 2.6. Its pending state is 2. Calling the query or reaching the end
of a file does not silently trigger that future timeout. A separate decoder
could inspect pending state, but that would be a declared query operation.

Moving C earlier or later can change its membership. In a richer unit,
changing content can change its computed arrival, which changes the group,
which changes emitted content and future state. Within a fixed membership
history, derivatives teach timing/content transformations. Crossing a hard
membership boundary requires credit for the alternative complete outcome.
Simply differentiating the old group's sum misses that discrete consequence.

The next layer can receive the emitted content **and its emission time**.
With a temporal mode exp(−rho*age), the different times also change the
representation at a later join/query. A persistent write can influence future
events long after this emission. Those interactions explain why route,
reception, content and memory credit belong in one model description.

These policies are family choices. The native fitted stack's default selects
one receiver per head; the window/silence references are not yet a complete
integrated native learner. The scalar demonstration does not charge discovery,
alternative training, optimizer or hardware cost and advertises no speedup.

Source contracts: [silence timeout](../sleeping_machines/silence_burst.py),
[race window](../sleeping_machines/race_window.py), and
[smooth temporal-window moments](../sleeping_machines/temporal_window.py).
