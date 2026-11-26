# Broad original-text coverage preflight

Actual dry-run completed; **training NOT started**. The current four-modality
>=95% / <=200M-before-quantization objective remains unmet.

## Measured coverage

The original training corpus contains23319 questions. The inherited all-block
run and recent deterministic control use the same8192 original question IDs.
Conservatively exclude every original ID in their full run configurations,
including updates beyond a selected checkpoint.15127 original IDs remain.
All these records were already used by the earlier last-block-only adaptation;
this is new coverage for subsequent all-block training, not lifetime-new data.

The actual seeded sampler generates2048 batches of8 questions:4 original from
the remaining pool plus4 curriculum.16384 distinct IDs, no within-run repeats.
Original coverage is8192 questions/5059 states:1733 score,3238 noul,3221 choice.
Curriculum reuses8192 previously trained questions; preserve this limitation.
Immutable corpus hashes and both source configurations are recorded in JSON.
Original/source ID and normalized-state split checks passed without opening audit data.
Order SHA256: `f32de459ec5ddf7d5d0ec89c012d2de512099028ad61e9ce9d065697b23828bb`.

This is a possible future data-coverage comparison, not a measured improvement.
It does not establish that more examples solve the quality gap. Before launch,
the initialization, validation-only selection, budget and retention policy must
be fixed, with a separate output and process. Existing runs remain unchanged.

## Storage gate failed

At measurement:621633536 free bytes. Required704643072 bytes:
160MiB output allowance plus512MiB reserve. The allowance covers two42.62MB
weights, a temporary best-weight replacement, saved predictions and logs.
No allowance was silently lowered and no historical file deleted.

Local evidence: `C:/CodexArtifacts/pretrained-reuse-r2/text-coverage-preflight-01/`.
It contains the full sample order and result. No model was loaded, trained or
checkpointed; no job was scheduled in the background. More storage is required
before an experiment with this retention policy can run safely.

Reproduction: `python -m scripts.plan_text_coverage` in the existing environment;
its output path is exclusive and deliberately fails if already present. Use a
new explicit path in a subsequent revision rather than replacing this evidence.
