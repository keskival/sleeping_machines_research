# Pilot kit: event-native anomaly detection on your own event logs

What a design-partner pilot needs and delivers. This is the route to TRL 6 for the EIC Accelerator
(`investment/EIC_ACCELERATOR/EIC_ACCELERATOR_PLAN.md`).

## What the partner provides

- An **event log export** (CSV) with:
  - `timestamp`: ISO 8601 or epoch seconds;
  - `event`: event type (state change, message code, alarm code, measurement category, ...);
  - optionally `stream`: machine, line, host, service or patient id.

  No payload text or personal data is needed; codes can be pseudonymised.
- A **reference period** the partner considers normal operation (weeks to months), plus the monitoring period to
  evaluate.
- Optionally, **known incidents** (time and stream) **for evaluation only**. They are never used in training.

## What the partner gets

- Per-window anomaly scores with **calibrated alarms**: thresholds set on held-out normal data to a declared
  false-alarm rate, default 1%.
- For every alarm, the **event type that drives it**: the step whose timing deviates, the *slowdown* statistic of
  THEORY §440.
- A report: alarms, false-alarm rate on reference data, detection lead time against known incidents.
- Everything runs on ordinary CPUs and can run on premises.

## Pre-registered success metrics (agreed before scoring)

- Detection rate of known incidents at the agreed false-alarm rate.
- Median lead time before the incident's recorded onset.
- Comparison with the partner's current rules or detector on the same windows.

## Running it

```
python experiments/pilot/event_anomaly.py --csv log.csv --stream-col machine --window-s 21600 \
    --reference-until 2026-09-30T00:00:00 --out pilot_run/
```

Outputs:
- `scores.csv`: window, `mean_nll`, `glr_max`, driving event, alarm flags;
- `report.json`: data summary, thresholds, alarm counts, model size;
- `model.pt` and `vocabulary.json`.

## Method

- **Model:** a race-of-delayed-clocks marked point process. It is the model that beats the published state of the art
  on three EasyTPP datasets (`experiments/B1_EASYTPP.md`).
- **Timing resolution:** detected from the data; the model never resolves time below the recording resolution (THEORY
  §439).
- **Statistics:**
  - mean negative log-likelihood (general typicality);
  - the per-event-type slowdown GLR (THEORY §440.3), the locally most powerful family against steps that slow down.

## Validation

`experiments/pilot/synthetic_check.py` builds a synthetic plant log in which half of the machines develop a slowed
step after the reference date, runs the kit, and reports AUROC and alarm rates. See its latest result in the B3 log.
