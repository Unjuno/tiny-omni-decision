# Cross-process numeric training trace and deterministic control

Execution PASS for four bounded diagnostic processes,16updates each. First128
question encodings and all initial trainable tensors match between512- and2048-
step materialized plans. The trace includes actual AdamW updates, unlike the prior
same-input backward-only checks. No validation forwards or quality selection.

## Default execution

Separate processes, deterministic algorithms off and no explicit cuBLAS workspace:
steps1–2 match loss,gradient norm,all gradient/updated-parameter hashes. At step3,
78 gradient tensors and47 updated parameter tensors differ, despite equal scalar
loss/norm. At step16,99 gradient tensors and97 parameter tensors differ; loss
0.9670063257 versus0.9670062661. This reproduces early cross-process divergence.
The exact low-level operation/kernel remains unidentified; do not blame only the
numeric embedding. Data preparation size also changes allocation/process context.

## Deterministic execution bundle

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization and enable
`torch.use_deterministic_algorithms(True)`. In two new separate processes, all16
losses,norms,gradient hashes and updated parameter hashes match exactly, despite
different materialized budgets. Both trace JSON files SHA256:
`4b0f9f35de40ed139e441daf9e2c0612587025e1538a7a4d703303e9c24f15e5`.

This validates the bundle on this bounded trace, not the complete512/2048 history.
It changes numerical execution from old runs; do not present it as exact resume
or silently require equivalence to a nondeterministic historical trace. The
stopped long01 remains incomplete and is not overwritten.

## Next experiment requirements

Use a newly declared deterministic control/candidate pair, same seed/initial
checkpoint/data/order/budget, differing only numeric-row adaptation. Preserve old
results as reference. Verify a short cross-process prefix, record runtime flags,
and retain strict within-new-protocol reproducibility checks. Any mismatch must
save detailed records and stop. Do not loosen historical gates or replay until
lucky matching occurs. No additional seeds, transfer tuning or audit access.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/numeric-prefix-trace-*`;
no training weights adopted/saved, only hashes and traces.64total diagnostic
optimizer updates, no claim of quality improvement. Main initial validation is
not run in this harness, so its allocation effects are not reproduced.
Ruff passes;399CPUtests pass,1skip. Four-modality95–98% remains unmet.
