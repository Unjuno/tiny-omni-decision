# Path A VideoMAE Clevr-4 image Decision results — 2026-11-13

## Result scope

This records one fixed, seed-17 frozen-feature Decision experiment. It tests a
closed-set synthetic **static image taxonomy** task only. It does not establish
natural-image understanding, open-ended question comprehension, temporal video
reasoning, text/audio/video input, multimodal fusion, quantization behavior, or
the product target. The one-time reporting partition is internal development
data, not a blind audit or a product release evaluation.

## Frozen inputs and model

- Source commit: `519ba902e9cc61b8c00d9f3933fae23cd9fe9b15`.
- Frozen input directory: `D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03`.
- Input receipt SHA-256: `350e7d4dec9f113d941f1415dc3bda4e556315ea57baffb89fe2efa62717a16b`.
- Frozen config SHA-256: `97f32fd4a2172526c67265f7cd74a153c993fd2e05992ce0b4899de8a04f160c`.
- Feature manifest / media rows SHA-256: `019cb10f6943c1e6b58ff671215e657d310a65cd53162538c4e3c3789d0694f6`.
- Feature array: 78,306 × 384 FP32; SHA-256 `7b6cef6b6451eb7cafd8d918e34fe527fac20fb658ae79c692cac28c9130e4b1`.
- Inputs: 72,010 train images, 1,000 selector images, 5,296 disjoint one-time reporting images; all image assets are unique within their respective partitions. The 5,296 reporting labels were read only after selector checkpoint selection and best-checkpoint reload.
- The official archive size matched its pinned size (36,057,238,315 bytes). The whole archive SHA-512 was **not** recomputed in this run; every consumed image was checked against its SHA-256 manifest.
- Frozen encoder: OpenGVLab VideoMAE V2 ViT-Small K710, 22,153,286 parameters; source SHA-256 `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`; checkpoint SHA-256 `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- Image preprocessing: RGB Lanczos resize to 224, `/255`, mean/std 0.5. Each static image was repeated over 16 frames; **there is no temporal evidence in these features**.
- Decision readout: taxonomy-conditioned MLP, 52,682 trainable parameters. Combined pre-quantization model count: 22,205,968. Best readout: 212,985 bytes; encoder plus readout payload: 44,547,594 bytes, excluding runtime/dependencies.
- Fixed fit: seed 17, 8 epochs, 288,040 task examples per epoch, AdamW, LR `1e-3`, weight decay `1e-4`, constant scheduler, batch 512, no early stopping. Best checkpoint rule: minimum equal-taxonomy selector NLL, earliest epoch on ties.

## Execution record

Feature extraction ran on the local NVIDIA GeForce RTX 3080 Laptop GPU, Python 3.11.9, PyTorch 2.6.0+cu124, FP32, batch 8. It completed 78,306/78,306 rows from 2026-11-13 04:13:17 to 05:45:41 UTC (5,544.73 seconds; 92.41 minutes); peak CUDA allocation was 1,234,878,464 bytes. The encoder remained frozen. Source files were not modified and sealed audit data was not loaded.

The first readout process (attempt-01) stopped at the first backward pass because deterministic CuBLAS requires `CUBLAS_WORKSPACE_CONFIG` on CUDA 10.2+. It produced no checkpoint or metric. The same fixed recipe was rerun in a new output directory (attempt-02) with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python startup; no data, model, seed, or training setting changed. Attempt-02 completed all 8 epochs, reloaded the selector-best checkpoint with exactly matching selector NLL, and wrote the one-time report. It used the same local RTX 3080; fitting plus selector/report evaluation took 24.99 seconds, including about 21.14 seconds for fitting, with 73,258,496 bytes peak CUDA allocation.

Successful command (from the staged Path A worktree):

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -c @'
import sys
sys.path[:0] = ['src', 'C:/CodexArtifacts/pretrained-reuse-r1/python-deps']
from scripts.train_videomae_clevr4_option_scorer import main
sys.argv = [
    'train_videomae_clevr4_option_scorer', '--inputs',
    r'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-inputs-20261110-attempt-03',
    '--features',
    r'D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-features-20261113-attempt-01\frozen-features.npz',
    '--output',
    r'D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-readout-20261113-attempt-02',
]
main()
'@
```

External artifacts (not committed):

- Features: `D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-features-20261113-attempt-01`.
- Successful readout: `D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-readout-20261113-attempt-02`.
- Failed startup attempt is preserved at `D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-readout-20261113-attempt-01`.

## Learning curve

Train accuracy/loss are rolling metrics over shuffled batches, not a fixed
training-set evaluation and not a generalization-gap estimate.

| Epoch | Train rolling accuracy | Train CE | Selector macro accuracy | Selector macro NLL | Brier | ECE-15 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5220 | 1.3694 | 0.6820 | 0.8828 | 0.3979 | 0.0850 |
| 2 | 0.7302 | 0.7412 | 0.7430 | 0.6812 | 0.3192 | 0.0547 |
| 3 | 0.7664 | 0.6213 | 0.7620 | 0.6118 | 0.2917 | 0.0377 |
| 4 | 0.7822 | 0.5678 | 0.7770 | 0.5734 | 0.2752 | 0.0368 |
| 5 | 0.7928 | 0.5347 | 0.7823 | 0.5486 | 0.2664 | 0.0386 |
| 6 | 0.8003 | 0.5112 | 0.7850 | 0.5316 | 0.2595 | 0.0347 |
| 7 | 0.8054 | 0.4941 | 0.7888 | 0.5156 | 0.2527 | 0.0313 |
| 8 (selected) | 0.8102 | 0.4800 | 0.7963 | 0.5082 | 0.2495 | 0.0277 |

All selector macro metrics improved through epoch 8. This fixed eight-epoch run
does not show a validation plateau; it does not prove that more fitting would
continue to help. No extra epochs or seed were launched.

## One-time internal reporting results

The reporting partition has 5,296 images per taxonomy (21,184 decisions total).
Macro metrics weight the four taxonomies equally.

| Task | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Color | 0.9870 | 0.0461 | 0.0217 | 0.0126 |
| Texture | 0.7904 | 0.5980 | 0.2842 | 0.0435 |
| Count | 0.4343 | 1.3152 | 0.6585 | 0.0210 |
| Shape | 0.9889 | 0.0342 | 0.0171 | 0.0081 |
| **Macro** | **0.8001** | **0.4984** | **0.2454** | **0.0213** |

The large task gap is material: the representation/readout predicts color and
shape well on this synthetic split, texture moderately, and count poorly.
Calibration ECE alone does not imply useful probabilities when count accuracy
and NLL remain weak. The product's 95–98% objective is not met across tasks.
The older text-free CLEVR-4 probe reported 88.30% macro Accuracy and 58.19%
count Accuracy with a different frozen representation and split. This run is
not a matched comparison; the numerical difference does not establish that
VideoMAE is worse than VideoPrism or that the encoder alone caused the count
errors.

## Artifacts and integrity

The successful run report is
`D:\CodexArtifacts\pretrained-reuse-r2\clevr4-videomae100k-readout-20261113-attempt-02\run-report.json`.
Its SHA-256 is `a26a7b8a08d4d023d80eed44891255f3282d6cfe26ae27c024950669ff3b7f3f`.
It records the feature/config/receipt hashes, selector and reporting metrics,
all epoch metrics, checkpoint hashes, parameter counts, timings, VRAM,
`sealed_audit_loaded: false`, and `source_files_modified: false`. The best and
final readout weights are separate; the best checkpoint SHA-256 is
`437e920e7cda02ff2e28f19c96433af04aab3d70494030b6e3278809aa1cc39e`, and the
final checkpoint SHA-256 is
`1d6ff294efcf77cdd7448f6d64cf02382fb605ca21a761384fd3683733107046`.
Selector and one-time reporting predictions are separately saved and hashed in
the run report.

## Interpretation and next gate

This is positive evidence that a small readout can reuse the pinned frozen
VideoMAE representation for some closed-set Clevr-4 visual attributes, while
the weak count result identifies a specific limitation of this tested
representation/readout/data path. It does not isolate whether count failure is
caused by the encoder, image preprocessing, the readout, or the task labels.
Because this experiment contains no temporal signal, it says nothing about
VideoMAE temporal reasoning. It also does not provide evidence about text,
audio, or video Decision quality.

Do not tune on the one-time reporting subset, promote this artifact, or infer a
product quality result. The immediate next work is to add this evidence and its
limitations to the R2 record; any new model comparison needs a fresh, justified
task gate and must preserve the frozen inputs and this run unchanged. Sealed
audit, ternary quantization, other seeds, and additional training were not used.
