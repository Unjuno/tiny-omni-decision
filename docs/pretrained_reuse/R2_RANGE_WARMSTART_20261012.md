# Predeclared learned-initialization control

One bounded local optimization control: initialize the unchanged numeric-range
training recipe from the preserved narrow-format encoder-coverage best step2048
(checkpoint SHA4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318).
New AdamW, not exact resume. Same seed17,2,048additional updates,4+4question batches,
constant2e-5LR,.01weight decay,clip1,dropout off,all six encoder blocks/classifier
trainable, embeddings/pooler frozen. Same fixed source corpus hashes, sample IDs
and order, original and narrow-curriculum validation IDs/order. No tokenizer,
model size, data or sampling change. Total22,713,601parameters unchanged.

Compare first evaluation against the pinned parent history for both validation
corpora, then keep the same earliest minimum mean-NLL selector including step0.
Step0 may remain best; preserve it rather than selecting later by a different
metric. Run the declared budget unless numeric/integrity/resource safety fails.
Keep best/final separately; verify exact best reload and frozen hashes.
Separate2-update smoke precedes the full run. New outputs range-warmstart-smoke-01
and conditional-repair-range-warmstart-01 under C:/CodexArtifacts/pretrained-reuse-r2.

Interpretation: this tests practical reuse of learned computation, not equal
TOTAL compute. Parent adds2,048updates after the common epoch3 text adaptation;
this new stage adds another2,048. The common earlier adaptation had8,745updates
and69,957question consumptions. Parent consumed16,384questions; planned new stage
16,384. Total planned across these three stages:12,841updates/102,725consumptions.
These are not unique samples. Original IDs repeat; numeric-range examples retain
latent state IDs, and25% of curriculum rows have the unchanged x transform.
Do not conflate per-stage unique consumption with lifetime unique coverage.

No prior transfer set is rerun in this experiment. After selection, diagnose
training fit on the same predefined training slice if needed; do not use it for
checkpoint selection. Any subsequent transfer study requires its own frozen
protocol. No95%, general reasoning or multimodal claim follows from retention
or training fit. No extra seed, budget extension, paid compute, product promotion
or legacy overwrite is authorized by this protocol. Local estimate20minutes;
check disk and live GPU state before launch.

## Training-fit endpoint fixed before completion

Measure the final2048weight on the same seed83,128-state training slice once,
even if the retention selector prefers step0 or another checkpoint. Report this
as final training fit only; it cannot replace the selected checkpoint or establish
transfer. The new evaluator mode rejects incomplete runs, wrong initialization,
unconsumed IDs and a final-weight hash mismatch. It reuses the prior slice without
rerunning the earlier models or any transfer set. New output warmstart-training-fit-01.
