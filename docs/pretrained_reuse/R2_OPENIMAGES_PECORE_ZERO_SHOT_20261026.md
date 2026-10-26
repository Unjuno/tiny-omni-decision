# PE-Core zero-shot comparison on Open Images V6 — 2026-10-26

## Scope and fixed task

This is a no-training image/text option-ranking comparison on the same 168
available rows from the page-declaration-matched, previously exposed Open
Images V6 development validation cohort used by the TinyCLIP diagnostic. The
question, four options, target labels, ordering, prompt template, scoring, and
metric code were held fixed. This is an exploratory paired comparison, not
blind evaluation or a model-selection result. No official test or sealed data
was loaded.

The prompt was `a photo containing {lowercase option}.` PE-Core's scaled
OpenCLIP image-text similarities were softmaxed over the four supplied options.
No training, prompt search, calibration, or checkpoint selection was done. The
control conditions used the same mean-color neutral image and deterministic
zero-fixed-point image derangement seed as the TinyCLIP experiment.

## Model and result

- `timm/PE-Core-T-16-384`, revision
  `7fe539ed578ac49a1c2b4f946e4b0747704c825a`.
- Audited OpenCLIP 3.3.0 route; checkpoint SHA-256
  `477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`.
- 69,546,113 instantiated parameters; weights file 278,219,132 bytes. It
  exceeds the provisional 200 MB package envelope before runtime dependencies.
- Hardware: NVIDIA GeForce RTX 3080 Laptop (not used), Intel Core i7-12700H,
  32 GiB system RAM. Python 3.11.9, PyTorch 2.6.0+cu124, OpenCLIP 3.3.0,
  CPU only, four threads, batch size 8. The aligned, neutral, and deranged scoring forwards took 8.40,
  6.90, and 7.26 seconds. No GPU or cloud compute was used. Peak process RAM
  was not captured.

| Input condition | n | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Correct image and options | 168 | 42.26% | 1.9034 | 0.8652 | 0.2988 | 0.7188 |
| Neutral mean-color image | 168 | 7.74% | 1.8874 | 0.9872 | 0.3743 | 0.4517 |
| Deranged validation images | 168 | 33.33% | 2.0414 | 0.9373 | 0.3318 | 0.6457 |

The aligned top-1 agreed with the neutral control on 35/168 and with the
deranged control on 66/168 rows. This demonstrates that image identity affects
predictions in this setup, while the aligned task accuracy remains poor.

## Matched comparison with TinyCLIP

| Model | Parameters | Weights bytes | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|---:|
| PE-Core-T-16-384 | 69.55M | 278,219,132 | 42.26% | 1.9034 | 0.8652 | 0.2988 |
| TinyCLIP ViT-39M / Text-19M | 83.10M | 332,443,148 | 64.88% | 0.9902 | 0.4899 | 0.0942 |

The models made the same top-1 prediction on 83/168 rows and had the same
correctness on 106/168. PE-Core was correct when TinyCLIP was wrong on 12 rows;
TinyCLIP was correct when PE-Core was wrong on 50. TinyCLIP had higher accuracy
and lower NLL, Brier, and ECE on this fixed cohort. Per-target counts are only
13–18 rows; the full per-target metric table is in the machine-readable paired
analysis. This result does not generalize beyond this exposed object-presence
task and is not a fair capacity-matched comparison.

## Reproduction and saved artifacts

Effective settings are in
[`path_a_openimages_pecore_zero_shot.yaml`](../../configs/pretrained_reuse/path_a_openimages_pecore_zero_shot.yaml).
The evaluator and paired analysis are
[`evaluate_openimages_pecore_zero_shot.py`](../../scripts/evaluate_openimages_pecore_zero_shot.py)
and
[`compare_openimages_zero_shot_predictions.py`](../../scripts/compare_openimages_zero_shot_predictions.py).
Focused tests cover media hash/path checks and fixed-ID/option pairing.
The evaluator SHA-256 is
`df3ec76d68b71d8b7ba90ddabc72ccaae19c4a257b44cef2a2216186e3c3f1e1` and the
paired analyzer SHA-256 is
`60f3b59dd87a178380f03c63b1342823c3e419b80df127c217428bcaebb563f9`.

```powershell
$env:PYTHONPATH='C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-02-runtime\packages;.'
$env:HF_HOME='C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-02-runtime\hf-cache'
$env:HF_HUB_OFFLINE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
python scripts/evaluate_openimages_pecore_zero_shot.py `
  --tasks-manifest C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-support-audit-20261026\page-matched-validation-tasks.json `
  --features-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-videoprism-features-20261026-attempt-01 `
  --media-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-media-cache-20261026-attempt-01 `
  --media-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-media-cache-20261026-attempt-02 `
  --model-file C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-02-runtime\hf-cache\hub\models--timm--PE-Core-T-16-384\snapshots\7fe539ed578ac49a1c2b4f946e4b0747704c825a\open_clip_model.safetensors `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-pecore-zero-shot-20261026-attempt-01 `
  --batch-size 8 --image-controls
python scripts/compare_openimages_zero_shot_predictions.py `
  --pecore-predictions C:\CodexArtifacts\pretrained-reuse-r2\openimages-pecore-zero-shot-20261026-attempt-01\predictions.jsonl `
  --tinyclip-predictions C:\CodexArtifacts\pretrained-reuse-r2\openimages-tinyclip-zero-shot-20261026-attempt-02\predictions.jsonl `
  --tasks-manifest C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-support-audit-20261026\page-matched-validation-tasks.json `
  --output C:\CodexArtifacts\pretrained-reuse-r2\openimages-pecore-zero-shot-20261026-attempt-01\comparison.json
```

Fixed task-manifest SHA-256:
`6fbd56f0caebc77b1067dfffaadb444aaf51b4b01eabff3855c11a245f86f964`.
VideoPrism validation-row SHA-256:
`c2396b3974de91afea66424b201388fb6d47013098a211d31b0191bd0a9adfb7`.
Media-manifest SHA-256 values for attempts 01 and 02:
`81b225f38d311ffe94003b2419100b33f88ad484d7f0f4ab1f3f35877e048560` and
`b854195055b47201fe6b7c890abda032ae0360636563b38069637617e2d2dd3f`.

External run outputs remain outside Git:

- `result.json` SHA-256 `fd4c4627e7ae93218a03670884fe1b491259e6e5e95551d6ee66f93c1794f8fd`
- `predictions.jsonl` SHA-256 `8118776c8d644d2b669f4cfc5f3e735435c769782c8fcb5779bb57e0e06b5050`
- `receipt.json` SHA-256 `d1432eff2e4b84f609cbad96170a08858c1fc8ed7d0964dfe61247cc2f8e30c7`
- paired analysis `comparison.json` SHA-256 `7a9ed4be6cf66669d3985508d5f4cd3185592f82342827910fb37d4920f25456`

Open Images page declarations matched CC BY 2.0 and the recorded attribution
profile; individual image ownership was not independently verified. The
evaluation does not close the R2 image quality gate, prove general image
understanding, or advance the four-modality 95–98% objective.
