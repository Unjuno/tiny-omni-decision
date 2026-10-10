# Path A evaluation protocol

Status: protocol draft for execution, fixed before any Path A training. No final or sealed data have been opened for this work.

## Scope and task contract

Evaluate pretrained frozen encoders as inputs to a candidate-conditioned Decision scorer. For every example the system must score the supplied options and return a normalized probability vector in the original option order. Do not treat text similarity, AudioSet tagging scores or public model-card benchmarks as Decision task accuracy.

Initial source tasks, only under the existing project source policy:

| Modality | Candidate task | Source and existing policy | Caveat |
|---|---|---|---|
| Text | Single-winner decision over the source options | Open-Jev redistributable release, CC0; Typed Decisions Synth, MIT, as controlled synthetic supplementary data | Group on upstream source/state and normalized content. Synthetic labels are not factual task evidence. Anonymous Open-Jev access returned HTTP 401 on 2026-10-10; its pinned manifest remains unchanged. The first runnable text probe therefore uses Typed Decisions Synth only and must be labeled synthetic. |
| Image | Four taxonomy decisions per image | CLEVR-4, CC BY 4.0 with attribution | Split by image identity; taxonomy decisions from the same image stay together. Synthetic classification is not a natural-image claim. |
| Audio | Ten-way spoken keyword decision | Speech Commands v0.02, CC BY 4.0 | English keywords only; group by speaker/source identity and keep the official speaker-safe source splits. This does not establish general speech or environmental-audio understanding. |
| Video | Supported categorical descriptive decisions, with a distinct temporal subset reported | CLEVRER, CC0 | Group every question/choice from a video scene into one split. Existing normalizer omits explanatory, predictive and counterfactual question classes, so this path cannot claim those capabilities. |

No approved synchronized audio-video decision source is present in the current project manifests. Therefore do not claim joint audio-video reasoning from these separate unimodal tasks. Sensor input has no approved dataset or checkpoint in this inventory and remains untested.

## Split isolation

- Freeze exact source revisions and hashes before normalization.
- Train, validation and sealed audit are separated by the strongest source identity: text state/source, image file, audio speaker/source, or video scene. A group is assigned once; related rows, options and decisions inherit that group.
- Preserve official source splits where they are documented. If a source has no independent final split, hold out whole identities from the training partition with the deterministic group splitter; never move a group after looking at predictions.
- Enforce zero intersection for source ID, asset/media ID, parent question and normalized-content hash across train, validation and sealed audit.
- Save sample IDs and order hashes for every split. The sealed audit is not loaded by training, checkpoint selection, calibration fitting or iterative analysis.
- Reuse neither legacy Teacher v0 final evaluation IDs nor previously observed legacy final-test results to select a Path A model.
- When a source split or rights fact cannot be proven from the pinned manifest, fail closed for that source.

## Outputs and metrics

At each checkpoint, report each modality and each source separately:

- Accuracy and macro accuracy over tasks, including count and number of independent groups.
- NLL from the returned option probabilities.
- Multiclass Brier score, defined as the mean over examples of the sum over options of squared probability error.
- ECE using 15 fixed-width bins on maximum option confidence; also report the bin counts.
- Video additionally reports question subtype and number of independent scenes.
- Joint tasks are reported only after a legally approved synchronized source exists; report matched unimodal ablations and missing-modality conditions.
- Reliability intervals use 10,000 scene/source-group paired bootstrap replicates, fixed seed 17, and percentile 95% intervals. Resample groups, not individual questions. If fewer than 200 independent groups are available, label the interval low-power and report the actual group count.

The primary selection score is macro NLL across available modalities after the fixed option scoring rule. Use validation only. Report macro accuracy and minimum modality accuracy as guardrails. Calibration temperature, if fitted, is trained on a separately named calibration subset carved by source groups from training data; validation and sealed audit cannot fit it.

## Invariance and repeated-observation checks

- Repeat feature extraction for a fixed observation and compare hashes/tensor tolerances under the pinned model, preprocessing, dtype and backend.
- Permute each example's options with a fixed permutation seed, restore output order, and measure maximum probability-vector difference.
- For one cached observation and multiple questions, verify the visual/audio tower is called once while the readout is called per question. Record avoided tower calls and cache invalidation on changed media.
- Report OOD by source or taxonomy held out before training. Do not label random row splits as OOD.

## Resource and deployment accounting

For each encoder and complete candidate, save exact revision, parameter count, tensor bytes, raw artifact file bytes, dtype, license, source commit, preprocessing, peak memory, warm/cold latency p50/p95 and device/software details. Count all simultaneously resident encoders, tokenizer tables, projection modules, normalization parameters and Decision head. Separate load time from steady-state inference and cache-hit latency.

Path A training is limited to small readouts over frozen cached features. Log the cache manifest, feature hashes, storage bytes, extraction runtime, training examples, trainable parameter count and optimizer budget. No backbone unfreezing or large-scale distillation is allowed in this phase.

## Acceptance rules

- A plumbing smoke is at most 8–32 examples and only proves I/O, determinism, shapes, finite values and save/reload.
- Quality evaluation starts only after enough independently grouped records and zero-leakage checks are demonstrated.
- Freeze task thresholds and resource caps before seeing validation results. If a modality has inadequate sample coverage or only synthetic proxy labels, mark its product gate UNRESOLVED; do not pool it with another modality.
- No promotion based only on pooled accuracy, a single modality, an uncontrolled zero-shot similarity score or a public benchmark reported by the checkpoint authors.
- Open the sealed audit once, only after architecture, preprocessing, probability calibration, thresholds and code are frozen.
