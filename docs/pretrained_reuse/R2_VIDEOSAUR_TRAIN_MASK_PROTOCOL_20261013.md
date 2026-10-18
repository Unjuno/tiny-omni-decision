# VideoSAUR fixed train-mask transfer protocol — 2026-10-13

Question: does the actual natural-video checkpoint retain marked-object identity
well enough to justify a later Decision representation probe? This diagnostic
cannot establish Decision Accuracy, generalization or product admission.

- Same first eight train-manifest videos as StoSAVi diagnostic; no exclusions.
- Same all-consecutive-frames/exclusive-cut policy and exact unmasked annotation
  frames. No labels/questions/validation/sealed audit used for model inference.
- Full frozen93,530,201-parameter reconstructed VideoSAUR, exact checkpoint/config
  limitations from R1 runtime report. GPU FP32, seed17 per video.
- Seven initial slots sampled once on GPU; carry `state_predicted` after each
  frame. Initial correction3iterations only at absolute frame0;2thereafter.
- Encoder batches6; no temporal state reset at batch boundaries. Separate CPU
  synthetic9-frame check with chunks2/4/3 verified exact state and final predicted
  state against upstream ScanOverTime. This is recurrence logic validation,
  not numerical equality of all GPU encoder batch shapes.
- Tensor direct518x518 bilinear resize, antialiasTrue, ImageNet normalization.
  Decoder masks37x37 bilinear resized64x64 with align_cornersFalse for the existing
  fixed pixel-center geometry helper. No training or selection of preprocessing.
- Oracle frame0 highest mask/box IoU chooses a slot; keep its index. Record
  per-frame best-slot oracle and static initial-box controls. Same metric helper
  and annotation/media hashes as previous diagnostic.
- Save protocol before extraction, masks, frame IDs, per-video progress, full
  results, timings and peak VRAM. Fail on hash mismatch or nonfinite values.
- Same policy is an engineering comparison; model, resolution, decoder and device
  differ from StoSAVi, so do not attribute any difference solely to pretraining.
- Assess whether fixed identity and oracle localization both show useful signal.
  No automatic adoption/quality threshold is set from these eight development
  videos. A positive result requires a separate predefined Decision protocol;
  a negative result should stop expansion of this route, not trigger a seed sweep.

Run root: C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-masks-01/

Harness: C:/CodexArtifacts/pretrained-reuse-r1/videosaur-source-audit-01/train_masks_diagnostic.py

Parent repository commit: cdf5c98. Hashes/environment are in protocol and R1
runtime artifacts. Existing outputs remain protected. GPU was0MiB/0% at launch;
no process was stopped, no cloud charge. Original training non-shape settings and
weight redistribution rights remain unresolved as documented in R1.
