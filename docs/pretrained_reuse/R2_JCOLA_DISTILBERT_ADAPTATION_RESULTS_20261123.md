# Japanese DistilBERT source-grouped adaptation — 2026-11-23

## Result and interpretation

This controlled comparison asks whether adapting the upper two layers of a
budget-compatible LINE Japanese DistilBERT improves over training only a
shared option scorer on frozen representations. It used only the pinned
JCoLA in-domain TRAIN file and a fixed ten-fold article/source-held-out
protocol. It did not load publisher validation, OOD development, test, or
sealed-audit data.

The encoder adaptation learned some signal for the minority *unacceptable*
class, but the fixed epoch-5 candidate had lower sample accuracy and worse
probability quality than the frozen readout. Source-cluster bootstrap
intervals support degradation in accuracy and NLL; the macro-accuracy gain
is uncertain. This is evidence against this compact model and training recipe
as a useful JCoLA decision model. It does **not** establish that BERT-family
encoders are generally unsuitable, and does not support the four-modality
product target.

| Metric (6,919 out-of-fold examples) | Frozen four-layer encoder + scorer | Upper two layers adapted | Difference |
|---|---:|---:|---:|
| Accuracy | 0.8338 | 0.8208 | -0.0130 |
| Macro accuracy | 0.5000 | 0.5632 | +0.0632 |
| Unacceptable recall | 0.0000 | 0.1774 | +0.1774 |
| Acceptable recall | 1.0000 | 0.9490 | -0.0510 |
| NLL | 0.4560 | 0.5188 | +0.0627 (worse) |
| Brier | 0.2820 | 0.2895 | +0.0075 (worse) |
| ECE (15 bins) | 0.0316 | 0.0977 | +0.0661 (worse) |
| MCC | 0.0000 | 0.1821 | +0.1821 |

The frozen scorer predicted “acceptable” for every example, so its 83.38%
accuracy is the majority-class result, not evidence of useful discrimination.
The adapted scorer reduced that collapse but remained weak on unacceptable
examples (17.74% recall) and became substantially less calibrated.

### Per-source results

| Held-out source | n | Unacceptable share | Frozen Acc. | Adapted Acc. | Frozen unacceptable recall | Adapted unacceptable recall | Frozen NLL | Adapted NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gunji_1987 | 246 | 11.4% | 0.8862 | 0.8659 | 0.000 | 0.071 | 0.361 | 0.409 |
| Inoue_1976 | 1,426 | 14.2% | 0.8576 | 0.8450 | 0.000 | 0.074 | 0.446 | 0.518 |
| Kuno_1973 | 1,246 | 22.8% | 0.7721 | 0.7737 | 0.000 | 0.211 | 0.534 | 0.577 |
| Kuroda_1965 | 266 | 6.8% | 0.9323 | 0.8609 | 0.000 | 0.222 | 0.300 | 0.384 |
| Kuroda_1992 | 549 | 14.6% | 0.8543 | 0.8270 | 0.000 | 0.250 | 0.418 | 0.438 |
| Miyagawa_and_Saito | 470 | 17.7% | 0.8234 | 0.7915 | 0.000 | 0.193 | 0.468 | 0.580 |
| Shibatani_1976 | 1,788 | 16.3% | 0.8367 | 0.8294 | 0.000 | 0.130 | 0.448 | 0.530 |
| Shibatani_1990 | 311 | 9.0% | 0.9100 | 0.8360 | 0.000 | 0.214 | 0.375 | 0.388 |
| Tsujimura_1999 | 414 | 22.7% | 0.7729 | 0.7850 | 0.000 | 0.266 | 0.539 | 0.638 |
| Tsujimura_2013 | 203 | 19.7% | 0.8030 | 0.8571 | 0.000 | 0.450 | 0.471 | 0.404 |

## Uncertainty

A paired percentile bootstrap resampled the ten article/source groups with
replacement 10,000 times (seed 17), keeping each group's paired predictions
together. Adapted-minus-frozen 95% intervals were:

- Accuracy: `[-0.0281, -0.0032]`
- Macro accuracy: `[-0.0393, +0.0040]`
- NLL: `[+0.0390, +0.0795]` (positive is worse)

There are only ten clusters, so the bootstrap is a diagnostic with limited
resolution, not a confirmatory population guarantee. These are internal
source-held-out development results, not blind or sealed evaluation.

## Frozen experiment specification

- Dataset: `osekilab/JCoLA`, source commit
  `736d9eef3af04bb17e4cf59adf01325973234ff9`; only
  `in_domain_train-v1.0.json` was read (6,919 rows; SHA-256
  `232787aa7bb11f3c42e455aad59d5f54cf1ea8255e9de79f77840de60e0541bf`).
- Model: LINE Japanese DistilBERT, pinned revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`; weight SHA-256
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
- Ten source-held-out folds, seed 17; 5 epochs per arm, batch 16 decisions,
  same prompt/options and 128-token pair cap; no early stopping, model
  selection, class weighting, threshold tuning, or prompt tuning.
- Frozen arm: first four pretrained layers frozen; a 98,561-parameter shared
  scorer trained with AdamW `lr=2e-3`, `weight_decay=1e-4`.
- Adapted arm: same four-layer model; layers 0–1 frozen, layers 2–3 and the
  same scorer trained with AdamW `lr=2e-5`, `weight_decay=0.01`, gradient
  clip 1.0. Fixed final epoch 5 used, with 14,274,305 trainable parameters of
  54,010,625 total candidate parameters.
- Effective runtime: Python 3.11.9, PyTorch 2.6.0+cu124 / CUDA runtime 12.4,
  Transformers 5.6.2, SentencePiece 0.2.0, fugashi 1.5.2, unidic-lite 1.0.8.
- Local RTX 3080 run: 1,777.1 sec total (29 min 37 sec); adaptation took
  1,699.1 sec over the ten folds. Manifest records 898,064,384-byte CUDA
  peak allocation at the final fold; `nvidia-smi` showed about 1.1 GiB while
  this run was the only active GPU process. No cloud compute or charge.
- Run source commit: `5ac4a02424b0c03f71cf8a206be8f6b5a4cdc12a`;
  training script SHA-256
  `32fa6bea91defb4b1193c16d7a1f3ef87e32a1a61397f0893088da3190eceb81`;
  effective config SHA-256
  `baa419a85a710a1c104f7a34c7768881d925294a48b2f712b8b695aa73cef223`;
  fold assignment SHA-256
  `e2a9289ed8465ed4b080f9d3a7451b1809c02dc6b01bfcdd1a741c018fd7892c`;
  OOF prediction SHA-256
  `ea68dfdf9527fb466c77d18ebbad179bf559f024a05a45fc3ed0acaa76215fc4`.
- Run began `2026-11-23 07:49:10 +09:00`, finished
  `2026-11-23 08:18:47 +09:00`, status
  `COMPLETE_SOURCE_GROUPED_CV`. The ten per-fold safe-tensor files total
  574,958,720 bytes and remain outside Git.

The first launch was a separate setup failure because the environment lacked
Japanese tokenizer dependencies. It started no fold and is retained as
attempt-01 with `SETUP_FAILED` status. The complete run is attempt-02, using
only pinned additive tokenizer dependencies. No existing model, data, run,
or checkpoint was modified.

## Reproducibility artifacts

External run directory:
`C:\CodexArtifacts\pretrained-reuse-r2-local\jcola-distilbert-adaptation-cv-20261123-attempt-02`.
It contains effective config, fold assignments, every fold's epoch history,
final adapted and frozen scorer weights, per-fold and global OOF predictions,
run manifest, and `analysis-summary.json`. The initial failed setup remains in
the distinct `...attempt-01` directory.

The protocol and implementation are in
[`R2_JCOLA_DISTILBERT_ADAPTATION_PROTOCOL_20261123.md`](R2_JCOLA_DISTILBERT_ADAPTATION_PROTOCOL_20261123.md)
and [`fit_jcola_distilbert_adaptation.py`](../../scripts/fit_jcola_distilbert_adaptation.py).

## Decision

Do not promote this model or claim that BERT solves the product's text task.
This run shows that upper-layer adaptation can break a frozen majority-class
collapse slightly, but under the fixed unweighted objective it trades away
accuracy and calibration for limited minority recall. It does not justify a
larger BERT family sweep on this benchmark alone. The next useful step is to
select a product-relevant text decision task with appropriate balanced labels
and fresh grouped data, or—if JCoLA remains strategically important—preregister
one cost-sensitive objective comparison on the same training-only grouping
while keeping these results as the frozen baseline. Do not revisit the exposed
OOD development or sealed audit data.
