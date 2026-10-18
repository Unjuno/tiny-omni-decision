# Fixed DINOv2 Clevr-4 public validation result — 2026-10-19

## Purpose and limits

This is a one-time evaluation of the already-selected 100k-trained readout on
the official Oxford Clevr-4 100k public `val` partition. The readout, option
prompts, preprocessing, and metrics were fixed before its labels were decoded.
No training, calibration, prompt search, checkpoint selection, or sealed-audit
access occurred. The result is public validation evidence, not a blind test or
a product-quality claim.

The evaluated artifact is the epoch-8 best checkpoint from the matched
seed-17 fit. Its SHA-256 is
`6f08e88014ca043570888264e857821eb71b13106d9a3673ea81240a192a0439`.
Its architecture and the 39,594,113 pre-quantization parameter inventory were
unchanged. All 19,988 public-val image assets were distinct from the 80,012
official-train assets used in the frozen train/internal-validation pool by
asset ID and SHA-256. They also had zero identity or media-hash overlap with the
older 10k Clevr-4 validation manifest. Every prediction was emitted once and
retained with its ID, target, candidate order, and probabilities.

## Metrics

There are 19,988 images and four supplied-option decisions per image, for
79,952 decisions. Each taxonomy metric covers all 19,988 images. Calibration
uses the existing 15-bin ECE implementation.

| Scope | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Macro across four taxonomies | 90.24% | 0.2444 | 0.1314 | 0.0109 | 0.9103 |
| Color | 98.93% | 0.0333 | 0.0169 | 0.0027 | 0.9872 |
| Texture | 94.42% | 0.1631 | 0.0812 | 0.0118 | 0.9558 |
| Count | 67.77% | 0.7737 | 0.4248 | 0.0280 | 0.7006 |
| Shape | 99.86% | 0.0075 | 0.0026 | 0.0010 | 0.9977 |

The same selected checkpoint scored 90.10% macro Accuracy, 0.2478 NLL,
0.1331 Brier, and 0.0120 ECE on the 8,002-image internal validation split.
The public-val differences are +0.15 percentage points Accuracy, -0.0034 NLL,
-0.0017 Brier, and -0.0011 ECE. This agreement is reassuring for this
synthetic attribute task. Count remains the clear weakness, and Texture is
below the provisional 95% task target. Neither the macro score nor the easy
Color/Shape tasks may conceal that gap. This does not establish natural-image
reasoning or a four-modality Decision model.

## Execution and integrity

- Source commit before this evaluation: `05097b2ddf943259e12a20b2589d5947a332ee6b`.
- Runner: `scripts/evaluate_dinov2_clevr4_public_val.py`; the official-val label
  reader decodes only the explicitly requested `val` records and leaves other
  split label values opaque.
- Runner SHA-256: `3601dc1bb47a893a80e24a5f2e618f5ee9d9cb3e0f8b865ddc69bdb101b9b897`.
- Command (the wrapper waited for cooperative GPU yield, then held its 3,200 MiB
  reservation while the child ran):

  ```powershell
  python scripts/run_with_hotgpu_guard.py --max-wait-seconds 3600 --reserve-mib 3200 --block-if-free-mib 14000 --min-child-free-mib 8000 -- C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe scripts/evaluate_dinov2_clevr4_public_val.py --archive C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-videoprism-v0\clevr_4_100k_v1.full.zip --media-hashed-dir C:\CodexArtifacts\pretrained-reuse-r2\clevr4-100k-index-seed17-media-hashed-v1 --historical-validation-manifest C:\CodexArtifacts\pretrained-reuse-r2\clevr4-full-coverage-seed17\val-image-manifest.jsonl --dino-model C:\CodexArtifacts\pretrained-reuse-r1\dinov2-small-audit-01\model --text-model C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-multiquery-runtime-bundle-v2\models\text-minilm-en --reference-readout C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-100k-readout-seed17-matched-attempt-02\best-readout.pt --output C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-public-val-fixedreadout-attempt-01
  ```
- The first wrapper invocation used bare `python` as the child executable and
  exited before GPU initialization because the wrapper requires a resolvable
  executable path. The retry with the already-recorded Python 3.11 executable
  ran successfully; no child process or artifact was created by that failed
  invocation.
- Fixed source archive size: 36,057,238,315 bytes. Its SHA-512 was previously
  verified as
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`;
  the full 36 GB archive hash was not recomputed in this run. The annotation
  member SHA-256 was `1e51a1ea77da85c7c9203b422dcd3c337f053981f4bd6848be7c41f2784556cc`.
- Official split counts were checked as 80,012 train and 19,988 public val.
  The val label manifest is SHA-256
  `f4761d1da4d8ac9d384a3dc5c38bca7c84f0564a22f5d964ecbb524c09eb8c84`; its
  media manifest is `fd62b31fed23e9c034ec01b06c264f8cc35f97274a1cfc1d4b398e733d12c4`;
  the image-ID order is
  `d289b6d0910344a633c9625c869e20a28601d37bf8c63a0652e65d7b56bda8e7`.
- DINOv2 weights SHA-256:
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
  MiniLM weights SHA-256:
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
  Reconstructed option embedding SHA-256:
  `90c42871b4c454b12b9ce1516fd8bec7394c40ce630f4d01c5ec9bf468f318e7`.
- Hardware: NVIDIA GeForce RTX 3080 Laptop GPU, 16,384 MiB; HotUGPU reserved
  3,200 MiB and granted the child with 13,015 MiB free. CUDA utilization and
  memory returned to 0% / 0 MiB after completion. Peak allocated VRAM was
  331,773,440 bytes. Feature extraction took 259.47 seconds; the recorded
  `result.json` interval was 262.28 seconds. Initial annotation/media hashing
  occurs before its `started_utc`, so total child wall time was longer (about
  five minutes by the execution trace). Local compute cost: $0.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2,
  NumPy 2.4.6, Pillow 10.4.0. The Windows PyTorch allocator warned that
  `expandable_segments` is unsupported; the run completed without OOM or
  non-finite values.
- The 79,952 saved prediction rows were independently re-read on CPU. All IDs
  were unique and complete, probability vectors were finite and normalized,
  and recomputed metrics matched the saved metrics within `1e-6`.

## Artifacts

The immutable run output is
`C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-public-val-fixedreadout-attempt-01`.
It contains the frozen run config, public-val labels and SHA-256 media manifest,
ID order, FP32 frozen features, all per-decision probabilities, and result JSON.

| Artifact | SHA-256 |
|---|---|
| `run-config.json` | `eaa2a731800fe96c3224dfbeb84703c38e589bd64eb395298fe9cce8801a6128` |
| `result.json` | `ee8cc9a9a259a831a04a9cbe2e5b429a1f1b35d7991416b5a208c352d41da284` |
| `public-val-features.npy` | `0f488f836c6a604530af766b093d6848ffdb105298aaee5e9bf4f0faa01faf9d` |
| `predictions.jsonl` | `29b8490cc9b78538f5d6556b9fcbd543467a064d5f8a1d57d80a76feb2ac1d11` |

The prior 100k training corpus, model checkpoint, and earlier attempts were not
modified. No quantization, cloud compute, or model promotion occurred.

## Next step

Keep this as a fixed image-path result. The data points to counting/spatial
representation as the next image bottleneck; however, after this public split
has been observed, it must not become a tuning set. Prefer an independently
licensed natural-image Decision task with a fresh split, or run a
predeclared representation-vs-readout diagnostic only on the existing frozen
features and previously observed internal data. Do not launch another fit on
these public-val labels.
