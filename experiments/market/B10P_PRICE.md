# B10-P — predicting BTCUSDT prices (owner: AWS; user direction 15 October 2026)

User direction: predict prices and other decision-relevant quantities, not only the six-class side × tick-direction
projection. The six-mark tape likelihood stays as an auxiliary output; the primary targets are return distributions.

## Targets and scoring

For every trade i and horizon H ∈ {10 s, 60 s, 300 s}: the log-return in basis points from trade i's price to the last
trade price at or before t_i + H. Each model issues a full predictive distribution (3-component Gaussian mixture for the
neural models, Student-t for classical references) and is scored by the probability of the exact 0.01 USDT price cell
that the realized future price falls in: P(1e4 log((P_f − 0.005)/P_i) < r ≤ 1e4 log((P_f + 0.005)/P_i)). This is the
recording-cell principle of the time term applied to prices: a proper discrete likelihood that cannot reward
grid exploitation, identical for every model. Contract: the stable log-cell computation matches numerical integration
to 1e−13, including far tails where integration underflows.

Secondary (reported, not primary): directional accuracy on moves ≥ 1 bp, calibration of predicted volatility, and the
cost-inclusive P&L of the implied direction signal at 0/2/5/10 bp. A descriptive probe (15 October) found a linear
order-flow signal earns about 0.2 bp gross per 10 s decision, an order of magnitude below a 2 bp taker cost; a return
distribution that forecasts volatility and adverse selection is the useful product, and market making is the setting
where an edge could exist (order-book data required).

## Inputs (identical for every arm; causal)

Six-class mark and physical gap; log trade size; signed log tick move; log realized variance of completed one-second
returns over the past 60/300/1800 s; log trade count over the past 60 s; signed aggressor-quantity imbalance over the
past 10 s. Features are standardized with TRAIN moments. Implementation: `b10p_data.py`.

## Phase 1 (admitted 15 October): acquisition and fixed-weight development comparison

Same 801 TRAIN windows (Aug 1–24, stride 20) and 756 DEV windows (Aug 25–31, stride 10) as B10; DEV scored positions
128..1023. Each neural family warm-starts from its acquired seed-0 B10 tape model (`aws_b10_stable_{family}_s0`), with
the new feature input and return head initialized prediction-neutral; 20 epochs, AdamW 5e-4, gradient clip 1, DEV
selection on the summed return NLL every two epochs. Classical references fit by maximum likelihood on all TRAIN window
events: unconditional Student-t; realized-volatility-scaled t; linear-feature t (location and log-scale linear in all
seven causal inputs). Drivers: `b10p_fit.py`, `b10p_references.py`.

Admission declaration. Stationarity: non-stationary stream; phase 1 is the acquisition/selection stage and its
fixed-weight score is labelled as such, not the primary verdict. Fighter: the native race-of-clocks member that won the
preregistered October tape confirmation, with temporal memory layers (elapsed-time decay), addressed per-mark memory and
the clock race retained, plus the new return head. References: LSTM and Transformer encoders with identical inputs and
heads (B10 founder exception for market references), and classical volatility models. State: reset per window in phase 1
(the training protocol of the acquired models); persistent across the stream in phase 2. Restrictions: window reset and
fixed weights at DEV scoring in phase 1 only, to select the configuration before the online test. Resources: wall/CPU
time, parameters and peak RSS recorded; full arithmetic ledger added in phase 2. Decision: which encoder and head enter
the online price confirmation, and whether learned temporal models beat feature-matched classical volatility models.

## Phase 2 (next): online price forecasting and a preregistered October confirmation

Continuous stream, persistent state, every arm adapts online after each block; a horizon label is used for learning only
after t_i + H has passed (delayed revelation), and every forecast is scored before its outcome can change weights or state.
Preregister a fresh October day (not 1 October, whose tape was already scored) and all settings before acquiring it.
September stays sealed.
