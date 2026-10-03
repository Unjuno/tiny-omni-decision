# Teacher v1 quality phase

**Status: implementation and data freeze in progress.** Teacher v0's evaluation
was previously observed. Its metrics remain a legacy reference in
[`DURABLE_TEACHER.md`](DURABLE_TEACHER.md) and are not used for v1 selection.
This phase does not perform ternary conversion, Recovery LoRA training, or final
Teacher merging.

## Integrity rules

The v1 training command takes a train JSONL and a mandatory independent
validation JSONL. It has no evaluation input. It rejects the same train/validation
path and any sealed-audit path before model loading or corpus reads. Normal
training reads only `data/processed/durable-teacher-v1/train.jsonl` and
`validation.jsonl`.

CLEVRER grouping uses the underlying video identity, so every question from a
scene remains in one partition. Images group by media identity; Speech Commands
audio groups by speaker ID where its `_nohash_` identity is available; Typed
Decisions Synth groups by source state; other text groups by normalized state.
Cross-split checks reject shared source IDs, source assets, media identities, and
normalized state/question/options/media fingerprints.

The sealed audit is built from candidate records after removing all record,
asset, media, and normalized-content identities seen in any Teacher v0 split.
It is written only under `data/sealed/durable-teacher-v1/`, separately from the
normal processed corpus. The ordinary corpus manifest stores its hash, not its
records. A selection lock freezes the selected adapter tree hash, exact config,
train and validation hashes, audit hash, seed, selected step, sampling policy,
base revision, and validation-selection rule. The evaluator verifies those
hashes, creates an exclusive one-time claim, and only then reads audit records.
An attempted evaluation cannot be retried, including after a runtime failure.
The selection lock binds both the sealed JSONL hash and its audit-manifest hash.

## Data policy

Only the ALLOW-approved sources below are eligible. Revisions and the source
catalogs are pinned in the manifests. MMAU, MVBench, OneJev REVIEW/DENY material,
and unknown-license media are excluded.

| Modality | Source | v1 source catalog |
|---|---|---|
| Text | Open-Jev; Typed Decisions Synth | [`manifests/teacher-v1-training-corpus.yaml`](../manifests/teacher-v1-training-corpus.yaml) |
| Image | Clevr-4 | same training catalog |
| Audio | Speech Commands v0.02 | same training catalog |
| Video | CLEVRER | same training catalog |

The held-out source catalog is
[`manifests/teacher-v1-heldout-corpus.yaml`](../manifests/teacher-v1-heldout-corpus.yaml).
Its validation and test source roles are explicit; sources marked `split` are
partitioned deterministically by their underlying asset group. Seed `17` is
immutable. Every output split and catalog is SHA-256 hashed. If a source loses
all eligible records after legacy exclusion, the manifests list the per-source
shortage and remaining asset counts. A train, validation, or audit split missing
an entire modality fails the freeze; observed legacy evaluation records are
never reused to fill a shortage.

Freeze the initial v1 corpora with:

```bash
tiny-omni-decision freeze-teacher-v1-corpus
```

The command writes train and validation under `data/processed/` and the sealed
audit under `data/sealed/`. It refuses to overwrite an existing v1 corpus.

## Experiment and selection policy

Seed `17` is the primary search seed. Start with the existing decoder rank-16
LoRA and frozen modality encoders/projector. The planned learning-curve budgets
are approximately 512, 1,024, and 2,048 optimizer steps, extending toward 4,096
only while validation improves. Actual schedules may be reduced if measured
local throughput makes the next checkpoint impractical. Every run, including a
failed attempt, has start and terminal events in its `experiments.jsonl` ledger.
Sampling policy/configuration comparisons use seed 17 and the fixed validation
score below. For a promising configuration, run seeds 19 and 23 when compute is
reasonable and report all three results, mean, and standard deviation. Those
replicas estimate stability; the final audit candidate remains the seed-17
checkpoint selected by validation, so a later seed cannot replace it merely
because its score is higher.

At scheduled validation points the trainer records train CE/accuracy aggregated
over every optimizer step since the previous validation, validation NLL,
Accuracy, Brier, ECE, per-modality metrics, overfit signals,
elapsed step/evaluation time, and the selected checkpoint. The selector minimizes
the predefined validation score:

```text
macro_nll
+ 0.2 × macro_brier
+ 0.1 × macro_ece
- 0.25 × macro_accuracy
- 0.25 × minimum_modality_accuracy
```

The checkpoint with the best validation score is reloaded; the last checkpoint
is not selected by default. Early stopping uses the configured patience and
minimum score improvement. Compare each candidate with Teacher v0 on the new
validation split by modality and reject a candidate that severely damages a
previously strong modality.

Sampling policies are explicit validation-only comparisons:

- balanced: `configs/decision/teacher_v1.yaml`
- weak-modality emphasis: `configs/decision/teacher_v1_weak_modalities.yaml`
- video priority: `configs/decision/teacher_v1_video_priority.yaml`

For each run, report configured weights alongside actual consumed counts,
unique examples/assets by source and modality, repeats, parameters, adapter
size, VRAM, and validation metrics. Prefer new unique samples over cycling a
small corpus. Capacity changes (rank 32, broader actual decoder targets, then a
small projector adapter) are considered one at a time only if the scaled rank-16
curves still show a validation capacity bottleneck. Full-base and modality
encoder fine-tuning are out of scope.

Model selection uses validation only and reports macro Accuracy, minimum
modality Accuracy, macro NLL/Brier/ECE, and each modality. Audio may not be
sacrificed to improve video. Once a candidate and all settings are frozen, run
the sealed audit once with:

```bash
tiny-omni-decision freeze-teacher-selection --candidate <selected-best-adapter> \
  --config-path configs/decision/teacher_v1.yaml \
  --train-path data/processed/durable-teacher-v1/train.jsonl \
  --validation-path data/processed/durable-teacher-v1/validation.jsonl
tiny-omni-decision evaluate-sealed-audit --candidate <selected-best-adapter> \
  --config-path configs/decision/teacher_v1.yaml \
  --train-path data/processed/durable-teacher-v1/train.jsonl \
  --validation-path data/processed/durable-teacher-v1/validation.jsonl
```

The target remains at least 90% Accuracy on **each** sealed-audit modality.
If a clean candidate falls short, report the plateau and likely bottleneck;
do not optimize against audit metrics, retry seeds for a better audit result,
remove difficult records, reduce options, or shrink the audit set.

## Frozen corpus and experiment results

Populate this section from the generated immutable manifests and experiment
artifacts before marking the phase complete.

- Train / validation / sealed-audit JSONL SHA-256: pending corpus freeze.
- Unique train examples and underlying assets by source/modality: pending.
- Pairwise record/media/content overlap: pending final corpus gates.
- Learning curves and attempted budgets: pending local runs.
- Selected rank-16 configuration and validation comparison to v0: pending.
- Sealed audit metrics: intentionally unavailable until validation selection is
  frozen; the one-time result will be recorded here.
- Local GPU/runtime/cost: RTX 3080 Laptop 16 GB; run-specific measurements
  pending. No cloud compute has been used.

The v1 artifact must be saved as `tiny-omni-decision-teacher-v1` only if it is
genuinely better on validation and passes overfit/forgetting checks. Never
overwrite the v0 artifact. The final report must include per-modality and
per-source Accuracy, NLL, Brier, ECE, mean confidence, data/checkpoint hashes,
compute, CI, and the evidence for continuing training or proceeding to
compression.
