# EgoViz fresh-session generalization evaluation protocol (2026-10-30)

## Purpose and boundary

Run a one-time development evaluation of the already-fitted linear and GRU
VideoPrism+MOMENT readouts on session groups that are absent from their train
and repeatedly viewed development corpora. Do not train on this set, choose a
checkpoint from it, tune a model against it, or call it a sealed audit. This is
a narrow video/IMU timestamp-correspondence diagnostic, not a practical action
Decision benchmark or a four-modality result.

## Discovery motivating the new set

The previously labeled 44-session aggregate is not a fresh validation set. Its
307 development sample IDs and all 20 session groups exactly equal the
development set used by the existing 773-example training-coverage and GRU
experiments. Its 24 training sessions are the same 24-session expansion in the
773-example training corpus. Its aggregate report explicitly says no quality
evaluation was run. Those files are retained unchanged and are not relabeled as
independent evidence.

## Frozen selection

- Source: official `humaidtech/EgoViz-120` clip index, SHA-256
  `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`;
  upstream card revision `7a84ecd3411b9f219b02f273c438f1ea12d74f52` states
  CC-BY-4.0 and defines `group_id` as the capture session. The card says clips
  in one group share the same worker, site and lighting, and explicitly
  recommends splitting on `group_id`.
- Previously used train/development groups: 48 + 20.
- Additional reserved groups: the prior plan's 16 previously used groups and
  20 prior sealed-audit groups. The sealed group IDs are used only to exclude
  records; no sealed media is retrieved, opened or evaluated.
- The exclusion union contains 104 groups. Of the remaining 76 groups, all 69
  with at least one clip of duration >=15 seconds and MCAP size <=1,000,000,000
  bytes are included. The other seven have no qualifying clip under these
  resource limits.
- One clip is selected per included group using a SHA-256 ordering of
  `seed=20261030` and chunk ID. No task label or annotation text is used for
  selection. The selected categories cover 9 of the source's 12 categories;
  the set is not a category-complete source audit.
- Frozen plan: `manifests/pretrained_reuse/egoviz-fresh-generalization-plan-20261030.json`.
  Planned cumulative downloads total 38,394,025,914 bytes; the largest single
  file is 972,987,959 bytes. Acquire and process one file at a time, require a
  3,000,000,000-byte free-space reserve before each acquisition, verify source
  size/hash and extracted feature receipts, then release only that newly
  downloaded raw file. Keep all prior corpora and checkpoints intact.

## Evaluation, fixed before acquisition

Use existing checkpoints only; evaluate on the new development groups once:

1. Linear readout at its fixed 5,632-update endpoint, to match the GRU budget.
2. GRU readout at its fixed 5,632-update endpoint.
3. The linear readout's historical step-616 validation-selected checkpoint,
   reported separately as the prior selector's choice.

Do not select among these from the new development metrics. Report
Accuracy/NLL/Brier/ECE, mean confidence, group counts, and category strata;
save per-sample IDs, targets, logits and probabilities. Use paired bootstrap
resampling by `group_id` for model differences and label the intervals as
development diagnostics. Preserve the same 8-frame VideoPrism and MOMENT feature
extractors, window construction, four-option task and target rule. No
publisher `/action` annotation is read. No sealed audit media is accessed.

## Limits

EgoViz publisher `group_id` is the documented session-level split unit, so
clips from one worker/session/site/lighting group do not cross the split. The
pinned card does not state whether a worker can appear under multiple
`group_id` values or provide a cross-session participant mapping; therefore
participant independence across groups remains unknown. This correspondence task has
timestamp-derived labels; it does not establish useful natural-world action
reasoning or prove either modality is necessary. Results cannot close R2-M or
support product promotion. Do not start additional training from these scores.

Primary source: [EgoViz-120 dataset card at the pinned revision](https://huggingface.co/datasets/humaidtech/EgoViz-120/tree/7a84ecd3411b9f219b02f273c438f1ea12d74f52).
