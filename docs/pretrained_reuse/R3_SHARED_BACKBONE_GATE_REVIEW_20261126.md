# R3 shared-backbone gate review — 2026-11-26

## Decision

**Do not start a new R3 Lens/backbone experiment on the present evidence.**
The already implemented Path A has a shared VideoPrism image/video route and
fits the provisional pre-quantization parameter envelope. The recent matched
four-modality result is mixed and below baseline, but it does not isolate a
backbone-sharing problem. A V-JEPA 2.1 candidate's checkpoint grant remains
unresolved. Preserve Path A as the reference and address the open R2 task and
data-quality gates before adding an unlicensed or unmotivated backbone.

## Evidence

- The Path A package audit counts **170,499,716 parameters** across VideoPrism,
  Whisper tiny, and MiniLM. The same VideoPrism encoder serves image and video
  input paths; the audited package does not contain a second image backbone.
- The current FP32 package is **687,254,772 bytes**. A storage-only,
  group-wise INT8 estimate is **182,018,983 bytes**, below the provisional
  200,000,000-byte envelope. This is an estimate only: no quantization,
  device load, or quality-preservation run was performed.
- The 2026-11-24 matched four-modality integration report found macro Accuracy
  0.2973 versus native baseline 0.3071, with Video +5.5 percentage points,
  Audio +0.4 pp, Image −8.8 pp, and Text −1.3 pp. It gives no controlled
  evidence that another video Lens will improve quality.
- The R1 inventory marks `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812`
  and its 86,833,152-parameter ViT-B checkpoint as `REVIEW`: the source-code
  licenses do not establish a checkpoint-specific product-use grant. Do not
  package or promote it pending an explicit grant.

## Gate interpretation

R3's prerequisite is a measured Path A inefficiency that a shared video
backbone and modality Lenses can plausibly address. The existing image/video
sharing means duplicate visual-backbone storage is not the diagnosed issue;
the remaining package gap is plausibly addressed by a separately measured
precision path after quality gates. The open performance issue is joint
Decision quality, and existing evidence does not attribute it to a lack of
shared layers. R2 remains open; R3 and R4 remain unstarted under their stated
gates.

This decision does not claim the Path A package is quantized, deployment-ready,
or that the 95–98% task goals are met. If later evidence shows a specific
distribution or capacity bottleneck that requires shared features, reopen R3
with a licensed checkpoint, matched A/B tasks, frozen data, and predeclared
activation/GPU-hour limits.

## Source records

- [Path A package-size audit](R7_PATH_A_SIZE_ENVELOPE_20261011.md)
- [R1 candidate inventory](R1_CANDIDATE_INVENTORY.json)
- [V-JEPA 2.1 checkpoint-rights recheck](R1_VJEPA21_LICENSE_RECHECK_20261030.md)
- [Cross-branch R2 evidence reconciliation](R2_CROSS_BRANCH_EVIDENCE_RECONCILIATION_20261126.md)
