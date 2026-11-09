# VideoMAE Clevr-4 100k Path A — fixed protocol

## Question and claim boundary

This is a first measured Path A image Decision experiment using a frozen,
pretrained visual representation and a small taxonomy-conditioned option
scorer. It asks whether VideoMAE V2 ViT-Small features transfer to four
closed-set synthetic Clevr-4 image-taxonomy choices on a newly frozen
internal-validation pool. It does not test natural-image understanding,
free-form language questions, audio, temporal video reasoning, multimodal
fusion, or the product's 95–98% target.

The input contract is an image, one typed task key (`color`, `texture`,
`count`, or `shape`), and the caller's allowed option strings. The scorer
returns a JSON distribution over those options in caller order. Task names
and class vocabularies are closed-set; this does not establish arbitrary
question comprehension.

## Frozen inputs and split evidence

- Source: Clevr-4 100k v1, publisher metadata `CC-BY-4.0`.
- Source archive: 36,057,238,315 bytes. The pinned preflight SHA-512 is
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`;
  this work does not recompute the whole-archive hash. Every consumed image
  is checked against its individual manifest SHA-256.
- Training: all 72,010 internal-train images, with four targets each.
- Checkpoint selection: 1,000 internal-validation images, fixed by the
  separate frozen receipt.
- One-time reporting: 5,296 disjoint internal-validation images. Readout
  selection must finish and reload before their labels are loaded.
- Quarantine: 1,706 of the original 8,002 internal-validation images were
  excluded by the frozen rule (prior ID collision or nearest pHash <=4).
- The scene-configuration audit found zero exact or 3-decimal-rounded scene
  collisions across train and internal validation. ID, asset, and media hashes
  are also checked across the exact experiment partitions before extraction.
- The official archive validation split and sealed audit are not used.

The current frozen labels/media files are authoritative. The older index
report contains stale labels/order hashes; this protocol uses the actual
current files whose hashes are recorded in the frozen input receipt.

## Model and fit

- Encoder: OpenGVLab VideoMAE V2 ViT-Small K710; 22,153,286 parameters.
- Exact source SHA-256:
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`.
- Exact checkpoint SHA-256:
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Encoder stays frozen. One static RGB image is Lanczos-resized to 224×224,
  normalized by `/255` and mean/std 0.5, then repeated for all 16 model frames.
  Thus this run has no temporal input signal.
- Feature width: 384, FP32. Feature cache rows are bound to ID and media hash.
- Readout: one 128-wide MLP conditioned by a learned 16-wide taxonomy
  embedding, outputting the fixed class vocabulary. It has 52,682 trainable
  parameters. Total exercised path: 22,205,968 parameters before quantization.
- Fixed recipe: seed 17; all 72,010 images; 288,040 image-task examples per
  epoch; 8 full epochs; batch 512 tasks; AdamW at `1e-3`, weight decay
  `1e-4`, constant LR; no early stopping. Select minimum equal-taxonomy
  selector NLL, earliest epoch on ties. Keep best and final checkpoints apart.
- No class-weight, seed, option-count, or prompt search is permitted in this
  run. No tuning on the one-time reporting subset.

The machine-readable configuration is
`configs/pretrained_reuse/path_a_image_videomae_clevr4_100k_seed17.json`.
Input materialization and validation are implemented by
`scripts/freeze_videomae_clevr4_inputs.py`; extraction by
`scripts/extract_videomae_clevr4_frozen_features.py`; fitting, checkpoint
selection, supplied-option JSON scoring and final one-time reporting by
`scripts/train_videomae_clevr4_option_scorer.py`.

## Preflight status

The run is staged in separate external directories under
`C:\CodexArtifacts\pretrained-reuse-r2`. The effective inputs are frozen at
`clevr4-videomae100k-inputs-20261110-attempt-03`; its config hash is
`97f32fd4a2172526c67265f7cd74a153c993fd2e05992ce0b4899de8a04f160c`.
Frozen input counts and zero cross-split ID/asset/media overlap are verified.
A separate CUDA smoke at
`clevr4-videomae100k-feature-smoke-20261110-attempt-01` on 32
actual official-train images produced finite `[32,384]` features on the local
RTX 3080 Laptop GPU, FP32, in 2.266 seconds excluding model load, with peak
allocated VRAM 1,234,878,464 bytes. The loaded model had the expected strict
checkpoint and 22,153,286 parameters. This is a smoke result, not a Decision
quality metric.

Full feature extraction and the fixed 8-epoch readout fit have not yet been
run. Do not describe this protocol or smoke as completed training or model
quality. Final reporting must include per-taxonomy Accuracy/NLL/Brier/ECE,
macro metrics, selected epoch, train/selector gap with rolling-train caveat,
unique assets and samples, compute/VRAM/time, saved prediction and checkpoint
hashes, and limitations.
