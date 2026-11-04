# DCASE2025 affine readout comparison

## Purpose and scope

The prior nested room cross-validation found that a width-64 GELU readout was
selected after a mean 1.1–2.1 epochs and retained positive train-to-validation
NLL gaps in most folds. This experiment changed one factor: the nonlinear
readout was replaced by one affine layer over the same frozen Audio/Video
features and 13-way event-class one-hot context. It tests whether the earlier
capacity aggravated cross-room overfit.

This is a development comparison on DCASE official dev-train only. It does not
access public dev-test annotations or sealed audit, does not select a product
candidate, and does not establish that this benchmark alone represents the
product task battery.

## Protocol and provenance

- Branch/source commit before local changes: `c3bfc1d7c325c445397939b75556b48dd870a531`.
- Feature cache: `C:\\CodexArtifacts\\pretrained-reuse-r2\\dcase2025-av-features-20261031-attempt-01`.
- Feature preflight SHA-256: `1be433e7f94a997e30b37caab89f74ee77477528c61437f61ce68d6917458524`.
- Query ID-order SHA-256: `7a963e607efbf688e7581c5473aa9e1910f30553775eb01895034c087518d901`.
- Effective config SHA-256: `a513d624cce3da4fc569cdef8601d830bb5c3b2d9752fe7de0444067ae1f9c2b`.
- Result SHA-256: `809522003354874c006e7a07db96edeb1b233172fe14db5772563112c175f8de`.
- OOF predictions SHA-256: `f46cf8b00c27bfb6baf0dca43382f64236d44f9f40a9b424a72eb50c850ca443`.
- Within-candidate paired bootstrap SHA-256: `f19456744ed30318a5ffbeac89a673fd18ced8b10223ba5770becda898d27daa`.
- Linear-vs-GELU exact-fold comparison bootstrap SHA-256: `844a465a63d591381a9d81862151e9949e018922fdc4ba55b521ccce6e67beba`.
- Paired comparison runner SHA-256: `a6b16950ba9f88b5d4f98b496dc0ed45be742ad1b48a46d517ec25d9d3a2579f`.
- Run-time runner SHA-256: `2582a0eec0646023ff1eb8a9bfef9f6781c1a3e2e797bf74de0073019ec6fbce`.
- Readout module SHA-256: `042d78133717010fc6306878a54b6dbcd1dd96fd9c51de0ac097a4cbfcce39b0`.
- Command: `python scripts/run_dcase2025_nested_room_linear_cv.py --features C:\\CodexArtifacts\\pretrained-reuse-r2\\dcase2025-av-features-20261031-attempt-01 --output C:\\CodexArtifacts\\pretrained-reuse-r2\\dcase2025-nested-room-linear-cv-20261101-attempt-01 --seed 17 --max-epochs 30 --patience 5` with `PYTHONPATH=.;src` and `CUDA_VISIBLE_DEVICES=''`.
- All 15,485 examples are outer-tested once by leave-one-room-out CV over 9 rooms; each outer fold selects by a different inner validation room. Parent recordings are disjoint across fold partitions. Standardization fits the 7 inner-train rooms only.
- Seed 17, batch size 256, CPU AdamW `lr=0.003`, `weight_decay=0.0001`; inner-room NLL selector, `min_delta=0.0001`, patience 5, maximum 30 epochs. All split, optimization and frozen-feature conditions match the preceding width-64 run; the affine head is the sole model change.
- Runtime: 36.76 seconds CPU for 27 fits (9 folds × 3 modes). GPU was not used; during the read-only machine snapshot the unrelated Orbitune process occupied the RTX 3080 Laptop at 15,757/16,384 MiB and 100% utilization. No process was interrupted. Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6.

## Results

Pooled outer-room OOF metrics; ECE uses the fixed 15-bin definition. The
event-prior baseline is identical in both runs and is estimated from each
fold's inner-training rooms.

| Arm | Readout | Accuracy | Balanced accuracy | NLL | Brier | ECE-15 | Onscreen recall | Trainable parameters |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Event prior | — | 75.44% | 50.00% | 0.5493 | 0.3656 | 0.0097 | 0.00% | 0 |
| Audio | GELU width 64 | 74.94% | 50.01% | 0.5793 | 0.3871 | 0.0692 | 1.03% | 50,178 |
| Audio | Affine | 71.28% | 50.14% | 0.6052 | 0.4089 | 0.0521 | 8.60% | 1,564 |
| Video | GELU width 64 | 73.01% | 57.21% | 0.5513 | 0.3607 | 0.0598 | 26.16% | 50,178 |
| Video | Affine | 68.25% | 60.29% | 0.6258 | 0.4193 | 0.0679 | 44.65% | 1,564 |
| Joint | GELU width 64 | 73.77% | 60.61% | 0.5418 | 0.3552 | 0.0387 | 34.74% | 99,330 |
| Joint | Affine | 69.26% | 60.54% | 0.5994 | 0.4036 | 0.0644 | 43.41% | 3,100 |

The affine head was selected at mean epoch 3.00 audio, 2.67 video, and 2.44
joint. Train-to-validation NLL gaps remained positive in 9/9 audio, 9/9 video,
and 7/9 joint folds (means 0.08, 0.06, 0.09). It did not eliminate
cross-room overfit. It raised onscreen recall and balanced accuracy for Video,
but substantially reduced raw accuracy and worsened NLL/Brier for all three
arms; the unchanged event prior remained better on raw accuracy and proper
scores. Do not adopt this smaller head as a quality improvement.

## Paired comparison to the width-64 GELU head

The saved OOF predictions were aligned by exact sample ID, room, parent
recording and target. The deterministic room layout was checked against the
GELU run's effective config and both runs' saved fold histories. A paired
cluster bootstrap used 10,000 replicates, seed 17; differences are affine minus
GELU. The initial exploratory bootstrap JSON is retained; the exact-fold
verified JSON is authoritative.

| Group | Arm | Accuracy difference (95% CI) | NLL difference (95% CI) | Brier difference (95% CI) |
|---|---|---:|---:|---:|
| Parent recording (78) | Audio | -3.66 pp [-4.80, -2.65] | +0.0260 [+0.0150, +0.0377] | +0.0218 [+0.0136, +0.0308] |
| Parent recording (78) | Video | -4.76 pp [-7.59, -2.15] | +0.0745 [+0.0427, +0.1090] | +0.0586 [+0.0350, +0.0854] |
| Parent recording (78) | Joint | -4.51 pp [-7.36, -1.90] | +0.0575 [+0.0291, +0.0891] | +0.0484 [+0.0254, +0.0744] |
| Room (9) | Audio | -3.66 pp [-5.36, -2.23] | +0.0260 [+0.0186, +0.0345] | +0.0218 [+0.0156, +0.0294] |
| Room (9) | Video | -4.76 pp [-11.49, +1.02] | +0.0745 [+0.0026, +0.1566] | +0.0586 [+0.0064, +0.1207] |
| Room (9) | Joint | -4.51 pp [-10.55, +0.44] | +0.0575 [+0.0106, +0.1201] | +0.0484 [+0.0058, +0.1027] |

The 9-room intervals are low-power. This single planned comparison does not
justify further tuning on these same rooms or a general conclusion about
linear versus nonlinear readouts. It does show that simply shrinking the
readout did not solve this dataset's cross-room transfer problem.

## Decision and next step

Retain the width-64 and affine runs as development evidence; promote neither.
The affine model is not a replacement for the earlier baseline. The strongest
signal remains room/domain shift and a label distribution where a simple
event-conditioned prior has 75.44% raw accuracy but zero onscreen recall.
Useful next progress requires a new admitted task/source or a cleaner target
that can be answered from both modalities, with group-disjoint data and a
predeclared evaluation. The exposed dev-test result remains untouched, the
sealed audit remains closed, and no existing v1/v2 checkpoints or feature
corpora were modified.
