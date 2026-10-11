# ViLT pretrained image/text fusion runtime audit

The frozen independent-tower token readout did not demonstrate reliable video
quality. Audit an existing jointly pretrained image/text Transformer before
further random-readout tuning. This is a new **runtime candidate**, not a selected
omni architecture or a reason to bypass the roadmap's conditional Lens gates.

## Provenance and load

- Author-hosted checkpoint: [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa).
- Pinned revision: `d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08`.
- Hub/model-card license: Apache-2.0. Card is written by Hugging Face, not the
  research authors; preserve this attribution. This does not license unrelated
  training images for redistribution or establish training-data decontamination.
- [Original implementation](https://github.com/dandelin/ViLT) and
  [paper](https://arxiv.org/abs/2102.03334) describe joint visual/language pretraining.
- Downloaded470,435,927-byte PyTorch checkpoint SHA256:
  `4d5f3409947b0369487ece7c5868f0040ceb67d25735dbb4ac5e99e03bab3a19`.
  Matches pinned Hub LFS metadata. All tokenizer/config hashes in companion JSON.
- Loaded with `torch.load(weights_only=True)`, no remote code or package upgrades.

Attempt01 downloaded successfully but strict load rejected a legacy persisted
`vilt.embeddings.text_embeddings.position_ids` buffer. Inspecting it confirmed
exact int64 `[0,...,39]`. A tested translation removes only this deterministic
buffer, requires exact values/shape/dtype, leaves the original dictionary and
checkpoint untouched, then requires strict load of all remaining tensors.
No learned weight is dropped. Attempt02 reuses the original local files read-only,
avoids another download, and saves results in a separate directory.

## Measured runtime

| Item | Measurement |
|---|---:|
| Complete loaded parameter count before quantization | 117,588,537 |
| Joint ViLT encoder | 111,595,008 |
| VQA classifier | 5,993,529 |
| Output | 1 × 3129 logits |
| Synthetic image tensor | 1 × 3 × 384 × 384 |
| Text length in probe | 8 tokens |
| First forward | 0.331792 s |
| Repeated forward | 0.056763 s |
| Repeated logits maximum absolute difference | 0.00001049 |
| Peak allocated VRAM | 491,298,816 bytes |

RTX3080Laptop16GB, FP32, Torch2.6.0+cu124, Transformers5.6.2. Synthetic image
with fixed red rectangle; no labeled task or accuracy metric. Repeated logits
are finite but not bitwise identical; deterministic equality is not claimed.
Load/processor time1.190022s. Retry field `download_seconds=0.816671` measures
metadata/local-hash checking, **not** the original470MB download, whose elapsed
time was not persisted before attempt01 failed. Local cost$0.

## Constraints and next gate

The shipped head has a closed3129-answer vocabulary. It is not a scorer for
arbitrary supplied options. Config has only40 text position embeddings; the
8-token smoke does not validate long state/question inputs. Check coverage and
lengths before any quality comparison; do not silently truncate or discard
unsupported options to raise accuracy. Video temporal reasoning and audio
capability are untested/unsupported in this checkpoint.

ViLT + existing VideoPrism would be231,953,721 parameters even before audio,
so that combination fails the200M total budget. ViLT + Whisper-tiny would be
155,349,177, but has **no established video path**. These are parameter sums,
not tested architectures. Replacing the classifier or treating images as video
frames requires separate design and evaluation; no product adoption here.

Artifacts:
`C:/CodexArtifacts/pretrained-reuse-r1/vilt-vqa-audit-01/model` (original files),
`C:/CodexArtifacts/pretrained-reuse-r1/vilt-vqa-audit-02/result.json` (successful audit).
Source `29fdced`; companion JSON records exact file hashes and environment.
Existing Teacher/features/runs remain unchanged. No new training, calibration,
validation inference or sealed audit access. Four-modality95% remains unmet.
