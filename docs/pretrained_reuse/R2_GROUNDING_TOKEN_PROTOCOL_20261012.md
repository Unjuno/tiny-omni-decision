# Descriptive token-readout protocol

The pooled descriptive probe has a small matched-scene signal; its independently
trained no-video difference is uncertain. Compare attention, mean and zero-video
modes of the existing VideoTokenReadout, with identical initialization, seed17,
AdamW1e-3/decay1e-3, batch8, 24epochs, earliest minimum equal-taxonomy validation
NLL including initialization. All modes have393,984 stored head parameters;
mean/zero do not use key/position weights. This contrasts readout mechanisms,
not perfectly equal active capacity. Do not compare causally against the previous
295,169-parameter MLP head without this limitation.

Inputs remain1,337 train/2,200 validation questions from256/200 disjoint scenes,
full variable options. Pad only for computation; remove padding before loss,
softmax and metrics. MiniLM and VideoPrism remain frozen. Re-extract exactly
456 token tensors into RAM, verify every token SHA and pooled parity against
existing references. No new media or labels, sealed audit or tower updates.

First run a separate two-epoch smoke with two questions per taxonomy per split.
Then run full24epochs per mode, retaining best/final and exact reload checks.
Record curves, all per-example probabilities, per-taxonomy/static-temporal
metrics, parameters, time, memory. After selection compare matched versus a fixed
scene-cyclic mismatch; no tuning to mismatched results. Report scene-paired
bootstrap with10,000 draws/seed17. No product adoption from this diagnostic.
Existing eight-frame token layout is unchanged; this does not itself test
reversed video or prove temporal reasoning.
