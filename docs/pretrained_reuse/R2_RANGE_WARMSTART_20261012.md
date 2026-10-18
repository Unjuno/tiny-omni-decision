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

## Completed result

Execution PASS: 2,048 updates; selected step 1408; exact saved-best reload and unchanged frozen-parameter hash verified. Original input hashes rechecked after completion. Total model22,713,601parameters; trainable10,647,169. No parameters added.

Local RTX3080Laptop run: 1122.35s excluding initial load/tokenization; peak allocated VRAM 3473190400bytes. No cloud compute. Source e7912b7; final-fit evaluator745e01d. Exact command/start UTC/config and every local file hash are in the adjacent JSON; result-file completion is after final reload, with elapsed duration recorded rather than an invented exact end timestamp.

| Step | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Selector |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.60236 | 0.87743 | 0.80566 | 0.46110 | 0.66926 |
| 128 | 0.61614 | 0.91327 | 0.77832 | 0.53267 | 0.72297 |
| 256 | 0.61063 | 0.90618 | 0.76562 | 0.54045 | 0.72332 |
| 384 | 0.61535 | 0.88603 | 0.78271 | 0.51683 | 0.70143 |
| 512 | 0.60197 | 0.92625 | 0.76025 | 0.56786 | 0.74705 |
| 640 | 0.60354 | 0.93678 | 0.79980 | 0.51108 | 0.72393 |
| 768 | 0.60354 | 0.96185 | 0.80371 | 0.49412 | 0.72799 |
| 896 | 0.59567 | 1.09275 | 0.80908 | 0.46454 | 0.77864 |
| 1024 | 0.59094 | 1.05334 | 0.81445 | 0.44124 | 0.74729 |
| 1152 | 0.59843 | 1.04317 | 0.79980 | 0.46159 | 0.75238 |
| 1280 | 0.60748 | 0.94427 | 0.81836 | 0.42541 | 0.68484 |
| 1408 | 0.60315 | 0.89998 | 0.82227 | 0.41314 | 0.65656 |
| 1536 | 0.59724 | 0.93205 | 0.80225 | 0.46921 | 0.70063 |
| 1664 | 0.61378 | 0.89754 | 0.81787 | 0.42203 | 0.65978 |
| 1792 | 0.59724 | 0.98586 | 0.83350 | 0.41184 | 0.69885 |
| 1920 | 0.60945 | 1.01902 | 0.83203 | 0.42804 | 0.72353 |
| 2048 | 0.60984 | 0.95081 | 0.83789 | 0.42065 | 0.68573 |

## Selected and final metrics

| Checkpoint / corpus | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| initial / original | 0.60236 | 0.87743 | 0.49486 | 0.01919 |
| initial / curriculum | 0.80566 | 0.46110 | 0.25546 | 0.03056 |
| selected1408 / original | 0.60315 | 0.89998 | 0.50757 | 0.06325 |
| selected1408 / curriculum | 0.82227 | 0.41314 | 0.23335 | 0.01736 |
| final2048 / original | 0.60984 | 0.95081 | 0.50977 | 0.08142 |
| final2048 / curriculum | 0.83789 | 0.42065 | 0.22733 | 0.03068 |

Whole stage: 16384consumptions, 16384unique question IDs and 6056state IDs. Selected checkpoint: 11264consumptions/unique question IDs and 5039state IDs. These counts are within this stage; they are not lifetime unique examples. Per-source/task counts are retained for every evaluation in JSON.

## Fixed final training-fit endpoint

| Task | Count | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 1024 | 0.83496 | 0.39565 | 0.21702 | 0.03161 |
| conditional_selection | 128 | 0.67188 | 0.82266 | 0.44795 | 0.12528 |
| count | 128 | 0.50000 | 0.99309 | 0.56467 | 0.09466 |
| extremum | 512 | 0.91211 | 0.22670 | 0.12207 | 0.02667 |
| temporal_order | 256 | 0.92969 | 0.22133 | 0.11762 | 0.04894 |

The fixed seed83 slice contains128training states/1,024questions, all consumed by the final checkpoint. Final fit83.50% vs27.25% for the earlier numeric-range run shows this training recipe can acquire the expanded values when initialized from the learned parent. It does not establish unseen-range/wording transfer. The parent contributes2,048additional inherited updates, so initialization and total training exposure are not equal-compute controls.

The selected checkpoint improves narrow-curriculum Accuracy80.57% to82.23% and NLL0.46110 to0.41314, but original-text NLL worsens0.87743 to0.89998 and ECE0.01919 to0.06325. Final original NLL0.95081 and ECE0.08142 are worse still despite slightly higher Accuracy. This is a measurable retention/calibration trade-off; no blanket absence-of-overfitting claim. Rolling train CE is not a fixed-example generalization gap.

No old transfer probe or sealed audit was loaded. No new multimodal quality or95% claim. The selected checkpoint remains a research artifact, not a product replacement. Next evidence needed: a separately frozen range-and-wording generalization design with equal-compute limits stated, plus continued image/audio/video reasoning work; do not silently extend this run.

Best SHA256: `f287bb0138fd0f264331cd2fc983567bc68323a85a25cda467387fffae7bc93b`. Final SHA256: `8392dc510aef5346d076028eb1b86cac3f631d6205671730e53b91264c2b7b6a`. Separate weights-only files, not strict optimizer/RNG resume. Legacy parent/base/corpus hashes verified unchanged. Artifact directory: `C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-range-warmstart-01`; fit: `C:/CodexArtifacts/pretrained-reuse-r2/warmstart-training-fit-01`.
