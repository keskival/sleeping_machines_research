# Calibrated local clipping adoption

Use fixed TRAIN-calibrated bank caps as the preferred clipping policy for future compatible native receiver-bank learners. The completed paired D8/seed6 continuation scores local6.226238129 versusglobal6.231192538 NLL (gain0.004954409), with measured single-thread process/scoring ratio1.004116. This is a positive locality and quality result; the quality gain is below the existing0.01automatic-promotion threshold and does not trigger an unchanged larger fit.

## Integration rule

- Reuse CalibratedReceiverBankAdamW and its exact bank-schema/checkpoint contract. Cover private, shared and output banks; preserve optimizer moments and bank versions.
- Calibrate caps using TRAIN gradients and the receiving learner's actual Adam moments. Freeze caps before quality fitting and charge calibration work. Preserve radial vector direction within each bank; do not replace this with coordinate saturation.
- Recalibrate when depth, bank schema, decoder, credit horizon, optimizer or data scale changes enough to change update geometry. The saved D8 caps are not universal constants.
- Carry the same policy into factual and counterfactual learners through the calibrated clone implementation. A global-policy teacher is not a valid local-policy consequence model.
- Confirm finite updates, schema coverage and checkpoint/resume behavior before using a new integration. Report completed prediction and whole-fit costs together.

## Current queues

Keep active source-bound language fits and the nine admitted B10 native/LSTM/Transformer globalclip1 comparisons unchanged. They answer already-defined questions. Native B10 uses ordinary full-model AdamW rather than the language receiver-bank learner; it needs a separately calibrated bank mapping before adoption. Dense reference architectures do not demonstrate our asynchronous substrate by switching their clipping rule.

After the three language repair comparisons complete, combine calibrated local clipping with the selected useful repair in the next authorized native development stage. Prefer the same-cut calibrated/global comparison when establishing transfer to a changed architecture. Do not launch a sweep across all models merely to change a default.

## Hardware scope

Each learning bank can limit its update without a training-time model-wide gradient-norm reduction. Offline calibration uses whole-model information. Existing factual differentiation, teacher scheduling and shared update boundaries still coordinate work. Measure useful independent bank execution and synchronization costs next; a removed clipping barrier alone is not a whole-trainer asynchronous speedup.
