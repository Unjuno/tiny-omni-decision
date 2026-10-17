# CLEVRER unique-scene coverage experiment v1

Protocol frozen before acquiring new clips or starting either training arm.

## Question

At a fixed 2,048 AdamW update budget, does adding 512 previously unused
CLEVRER training scenes to the existing 256-scene pool improve a frozen
VideoPrism + MiniLM Decision readout on the same 256 previously unused
validation scenes?

This is a single-seed development experiment. It does not establish general
video reasoning ability, a causal explanation for other benchmarks, or product
readiness. Validation is used for checkpoint selection and is therefore not a
blind audit set.

## Frozen data and models

- Dataset: `MIT-IBM/CLEVRER` at `98b842082ba4f7c18b6b9e3f39145871782a65ef`,
  CC0-1.0.
- Seed: 17. Each selected training/validation scene contributes one
  `static_descriptive` and one `temporal_descriptive` question.
- Baseline train pool: existing 256 scenes / 512 questions; read-only.
- Expanded-only train data: 512 new scenes / 1,024 questions.
- Shared validation: 256 new scenes / 512 questions.
- No sealed audit, official test, or previously used scene is loaded.
- VideoPrism: pinned model revision, 8 frames, 288px, frozen cached features.
- MiniLM: pinned revision, frozen encoder.
- Only the existing `CandidateScorer` readout is trained; both arms begin from
  identical seed-17 readout weights.

## Arms and controls

| Arm | Training scenes | Questions |
|---|---:|---:|
| Existing-pool baseline | 256 | 512 |
| Expanded coverage | 768 (256 + 512) | 1,536 |

Both arms use AdamW, learning rate `0.001`, weight decay `0.001`, batch size 8,
constant learning rate, exactly 2,048 updates, and evaluation every 128 updates.
There is no performance early stopping. The selected checkpoint minimizes the
unweighted macro NLL of the static and temporal question types, with earliest
tie and `min_delta=1e-5`. Best and final weights are saved separately.

The sampler shuffles the entire unique question pool at each full pass. If the
fixed update budget ends partway through a pass, the consumed counts are
recorded; examples are not duplicated within a pass. The two arms receive the
same update count, not the same number of passes or per-example exposures.

## Integrity and reporting

Before fitting, verify pinned source hashes, complete local media, unique scene
and media identities, zero train/validation scene and media overlap, cached
feature provenance and hashes, and identical ordered validation IDs for both
arms. Record normalized question-template overlap as a limitation rather than
claiming content-level independence. Save the exact effective config, source
hashes, per-step metrics, per-example predictions, selected/final checkpoints,
checkpoint reload parity, runtime/VRAM, unique coverage and a paired
scene-cluster bootstrap (10,000 resamples, seed 17, percentile 95% interval).

Primary outcomes are macro Accuracy, NLL, Brier and 15-bin ECE across the two
question types, plus each type separately. A favorable result supports only
the tested data-coverage change for this readout and this validation sample.
No conclusion about the multimodal product, 95–98% target, or architecture
ceiling follows from this experiment.

## Compute and preservation

Local RTX 3080 Laptop only; no cloud compute or paid services. All preexisting
corpora, caches, models, checkpoints and run directories are read-only. New
sample, feature, run and report files go under
`C:/CodexArtifacts/pretrained-reuse-r2/clevrer-coverage-512train-256val-v1`
or explicitly named sibling directories. This experiment does not load or
modify any sealed audit data.
