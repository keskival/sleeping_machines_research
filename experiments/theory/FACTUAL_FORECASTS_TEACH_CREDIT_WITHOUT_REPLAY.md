# Factual forecasts teach credit without replay

B10,13October2026. At an audited online decision, construct own/projected actual Adam proposals from the same actor parameters, moments and numerical state. Apply exactly one proposal. Hold its parameters fixed over the next32-target prediction window. The recorded factual losses are then the selected proposal's future losses; replaying this world again is unnecessary.

Let s be the selected proposal index, F_s its already observed mean next-window loss, and F_other the loss from replaying the alternative. The utility label remains Delta=F_own-F_projected: F_s-F_other if s=own, F_other-F_s otherwise. Cached factual losses are detached observations, exactly as the previous no-gradient replay labels were. The critic's stored activation and delayed update are unchanged.

Enforce equal actor parameters and equal pre-window numerical state against the selected proposal before reuse. No additional actor update may intervene within the window. The current fixed-version driver satisfies this condition; asynchronously delivered weight updates would require a version-bound prediction receipt and a revised contract. This is causally observed prediction feedback training the credit function, not BPTT-gradient imitation.

Acquisition uses exact BPTT actor updates rather than either candidate endpoint, so acquisition generally cannot reuse its factual world for these labels. Keep both replays there. The repair targets online deployment teaching only: two32-target future replays become one, while proposal generation, alternate forecasting, critic backward/Adam, key discovery and actor learning remain charged.

Smoke verifies selected-proposal parameter/state identity and unchanged saved forecasts/requests. Full16384forecast parity is required before retention. Nonlinear functions and arithmetic are separately audited for the remaining replay; an avoided execution count is not automatically an energy or complete fitting-cost win.
