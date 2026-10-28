# R1/R2 cross-branch evidence reconciliation (2026-10-28)

## Scope

Read-only review of current open PRs #16, #31 and #32 and their pinned
commits. No source branch, worktree, dataset, checkpoint, training run or
artifact was changed. The active RTX 3080 process was left untouched. The
findings below are development evidence and do not qualify a product model.

## PE-Core T16: real runtime and task result, but no promotion

PR #16 (`codex/r1-pe-core-audit-20261021`, head
`afea6624dfe247350ecbb704d61099304009f2a0`) records the pinned
`timm/PE-Core-T-16-384` revision `7fe539ed578ac49a1c2b4f946e4b0747704c825a`.
Its 69,546,113-parameter CPU OpenCLIP image/text smoke returned finite
embeddings. Selected converted tensors match the original checkpoint, but
full numerical parity and a save/reload output-parity test remain unverified.
The original implementation is under FAIR Noncommercial Research terms even
though the checkpoint/card and converted weights state Apache-2.0; this is a
material product-rights restriction pending legal review.

The frozen 131,329-parameter candidate scorer reached 79.95% macro Accuracy
on the already observed CLEVR-4 reporting subset, versus 88.06% for the
historical Path A predictions on the same IDs/options. Color, texture and
shape were strong (93.10–98.75%), but count was 29.28%, versus 61.19% for
Path A. The one-pass zero-shot scorer reached 42.81% macro Accuracy. These
results are synthetic, exposed-development comparisons; they do not justify
replacing Path A or sharing the PE-Core encoder. PR #16 is open and targets a
separate roadmap-status branch, so it has not been treated as merged evidence.

## DROID: new episode groups do not rescue video complementarity

PR #31 (`codex/r2-droid-site-universe-audit-20261028`, head
`aefb1a1a3cd09959e73dfac79dbf319fce63c367`) reports a metadata-only sample of
six official-train shards, totaling 6.25 GB. The six shards contain no new
collection site beyond the 14 already seen; this does not prove the full
release has no other sites. The measured transfers averaged 21.97 MiB/s across
six files (two three-file batches: 21.55 and 22.38 MiB/s). This rate is only a
rough estimate for similarly sized GCS shards and does not predict CPU feature
extraction time.

PR #32 (`codex/r2-droid-temporal-transfer-20261028`, head
`a8db60d262b9655cb00897f26cdec9ee6a177070`) replays unchanged readouts on 137
eligible, episode/group-disjoint records from later official-train shards.
All sampled sites were already represented in the earlier cohort, so this is
same-site group-disjoint development transfer, not an unseen-site test. On
these fixed records, video+sensor scored 39.42% Accuracy / 1.7790 NLL;
sensor-only scored 42.34% / 1.4968. Video permutation worsened joint metrics
in the earlier cohort, but that dependence did not produce a useful quality
gain over the matched sensor-only baseline. A post-hoc paired site-cluster
bootstrap of the saved later-shard predictions (10,000 resamples, seed 17,
14 sites) gives joint-minus-sensor Accuracy −2.92 pp (95% CI −8.08 to +0.97),
NLL +0.2822 (+0.1598 to +0.4871), and Brier +0.0887 (+0.0448 to +0.1595).
Accuracy uncertainty includes zero; the proper-scoring differences favor
sensor-only. This adds uncertainty quantification to existing predictions
only; it is post-hoc and not independent validation. Do not tune or refit on
either observed DROID validation/sample. Another DROID fit is not justified without a
prospectively specified task whose labels require visual state, and a new
evaluation cohort.

## Decision and next experiment gate

- Keep the current Path A and its measured task-specific readouts as research
  references; do not promote PE-Core or the DROID video+sensor readout.
- The PE-Core count failure and license restriction do not justify a shared
  encoder/Lens experiment. The DROID results do not justify another fit on
  those IDs or a claim that VideoPrism is generally unhelpful.
- The next quality experiment must first define a task where the claimed
  observation stream is causally needed, screen rights and group identities,
  then freeze source/feature/task hashes and unimodal/perturbation controls
  before fitting. For the current DROID action-direction task, visual-state
  necessity and a genuinely independent site remain unproven.
- No GPU training or download was launched in this reconciliation. PRs #16,
  #31 and #32 remain open and unchanged.

## Source records

- PR #16: <https://github.com/Unjuno/tiny-omni-decision/pull/16>
- PR #31: <https://github.com/Unjuno/tiny-omni-decision/pull/31>
- PR #32: <https://github.com/Unjuno/tiny-omni-decision/pull/32>
- PE-Core runtime/readout detail: `R1_PE_CORE_T16_CPU_AUDIT_20261021.md`,
  `R2_PE_CORE_CLEVR4_READOUT_RESULTS_20261021.md` and
  `R2_PE_CORE_CLEVR4_ZERO_SHOT_RESULTS_20261021.md` on PR #16.
- DROID site holdout: `R2_DROID_SITE_HELDOUT_VIDEOPRISM_SENSOR_RESULTS_20261028.md`
  on commit `23f02483866715a9aa52f42b6ca0625e987c0773`.
- DROID later-shard transfer: `R2_DROID_LATER_SHARD_FROZEN_TRANSFER_RESULTS_20261028.md`
  on PR #32.
- DROID later-shard site bootstrap: [report and output hash](https://github.com/Unjuno/tiny-omni-decision/blob/b8245103d221daea225f9666862f6b149462aa1b/docs/pretrained_reuse/R2_DROID_LATER_SHARD_PAIRED_SITE_BOOTSTRAP_20261028.md);
  derived on a separate branch from the PR #32 prediction SHA-256.
- DROID shard/site metadata audit: `R2_DROID_UNSEEN_SITE_SHARD_SAMPLE_20261028.md`
  on PR #31.
