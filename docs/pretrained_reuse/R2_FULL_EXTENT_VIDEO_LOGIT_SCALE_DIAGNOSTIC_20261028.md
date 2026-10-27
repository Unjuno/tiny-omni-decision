# R2 full-extent Video logit-scale diagnostic (2026-10-28)

## Question and scope

The completed native full-extent × fresh-coverage run improved several
non-Video scores with longer training, while its Video Accuracy stayed near
chance and its final Video NLL/calibration deteriorated substantially. This
CPU-only follow-up checks whether the saved Video logits show a confidence or
scale shift that explains that metric pattern.

This is a read-only analysis of existing saved development-validation
predictions at steps 512, 4,096 (the selected checkpoint), and 32,768 (the
terminal checkpoint). It did not load a model, train, fit calibration, choose a
checkpoint, change a prediction, or access sealed-audit data. The validation
set is exposed and was used repeatedly by the original selector; this report is
not a blind quality estimate.

## Pinned input and procedure

- Source run: `C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-fit-04/`.
- Run source commit: `dbdb2a89c39d41d6c351c2b7be471bdb13e2d259`.
- Run config SHA-256: `7758008edcfcedee819a6ed15b1f546f865568b2bfb56cc57a42ac2202138cb5`.
- Run result SHA-256: `dcc2190da3167d5fdaa2f30bb166a999d3ae0104d5a08e55cc1181f855c0df54`.
- Validation-ID SHA-256: `db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`.
- Original run status: `PASS_NATIVE_FULL_EXTENT_TRAIN`, terminal step 32,768,
  selected step 4,096, and `sealed_audit_loaded=false`.
- Video subset: 400 validation queries from 200 CLEVRER scenes. All three
  prediction arrays and the ordered validation IDs were checked against the
  run-result receipt before metrics were computed.
- Analysis commit: `93f12573a40b3112db8cd41acbe11ef721fb9413`.
- Script: `scripts/diagnose_native_video_logit_scale.py`, SHA-256
  `e8dab9eb2551dd95d639c0eef6c04434afa4b03d4809cd60698354f084653d85`.
- Command:

  ```powershell
  python -m scripts.diagnose_native_video_logit_scale `
    --run-dir C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-fit-04 `
    --steps 512 4096 32768 `
    --output C:/CodexArtifacts/pretrained-reuse-r2/native-video-logit-scale-diagnostic-20261028-attempt-01
  ```

- Output: `C:/CodexArtifacts/pretrained-reuse-r2/native-video-logit-scale-diagnostic-20261028-attempt-01/diagnostic.json`, SHA-256
  `318861478f8d13568be8e651b1c2dacc31c690b67b0fabcb737e9bbbcd8bf7f5`.
- Bootstrap: 10,000 paired resamples of scene-level mean differences,
  percentile 95% intervals, seed 17. Each scene is one resampling unit.

The script recomputes metrics in float64 from saved option logits. ECE here is
10-bin ECE; the full-fit report's evaluator uses its own documented metric
implementation/binning, so its values remain the primary run record. Small
implementation-level differences are expected.

## Results

| Checkpoint | Accuracy | NLL | Brier | ECE (10-bin) | Mean confidence | Median logit span | P90 span | Mean confidence when wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3475 | 1.3538 | 0.6997 | 0.0479 | 0.3896 | 0.5974 | 1.9051 | 0.3486 |
| 4,096 (selected) | 0.3375 | 1.3668 | 0.7223 | 0.1202 | 0.4543 | 0.7932 | 4.1617 | 0.4204 |
| 32,768 (terminal) | 0.3300 | 9.1019 | 1.2257 | 0.5931 | 0.9231 | 19.5913 | 54.1992 | 0.9152 |

From step 512 to 32,768, 248/400 predictions changed. Of 139 initially
correct queries, 72 changed to wrong and 67 remained correct; 65 initially
wrong queries became correct and 196 remained wrong. The net Accuracy change
was −1.75 percentage points.

Scene-paired bootstrap differences (32,768 minus 512):

| Quantity | Mean paired difference | 95% scene-bootstrap interval |
|---|---:|---:|
| Accuracy | −0.0175 | [−0.0750, 0.0375] |
| Mean top confidence | +0.5335 | [+0.5141, +0.5523] |
| Median-per-query logit span | +24.0382 | [+22.0076, +26.1520] |
| Top-two margin | +7.4021 | [+6.7101, +8.1241] |
| NLL | +7.7480 | [+6.6727, +8.8993] |

The Accuracy interval includes zero. In contrast, the intervals for confidence,
logit span, margin, and NLL are well above zero on this same selected
development cohort. The terminal checkpoint is therefore substantially more
confident and worse calibrated than step 512 without a demonstrated Accuracy
improvement.

## Interpretation and next gate

This supports a **Video logit-scale / calibration-drift diagnosis** for this
run. It does not prove why the scale grew, establish that longer training is
always harmful, or establish a capability ceiling for the backbone or model
family. The result is post-hoc analysis of the same development set used for
checkpoint selection, and the bootstrap does not remove that selection effect
or single-seed uncertainty.

Do not respond by fitting a temperature on these exposed IDs or by selecting
another checkpoint from the same audit-like cohort. Preserve the selected
step-4,096 model as the run's chosen checkpoint. A future calibration
intervention requires a separate training/selection protocol and a new
independent task cohort. Before any additional fit, retain the existing
recommendation to address Video label/task coverage and the train-probe versus
validation divergence with a prospectively frozen, rights-cleared task.

## Verification and integrity

- Prediction SHA-256 values: step 512
  `abceee5e02690ce01f726a72dc02869362e9bebeefe37848cbcc023f11da0a33`, step
  4,096 `f67e6b620a3d17fea5d390b8ca719367c0606c5c6623c665057e59956aefa996`,
  and step 32,768
  `6252321fbcf08459ecce717b1a0b23adc699ba9119c67358bd3fc09e3747d603`.
- Targeted CPU tests cover metric calculation, deterministic grouped bootstrap,
  and input-hash/audit-isolation validation.
- No source run file, model weight, corpus, or checkpoint was modified.
- No GPU time or cloud compute was used for this diagnostic.
