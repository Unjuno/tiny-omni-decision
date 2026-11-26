# CLEVR-CoGenT-B one-time compositional decision audit

**Status: complete, one evaluation only.** This is a project-held-out OOD
audit for one frozen image-plus-question option-readout candidate. It is not a
blind product evaluation, does not validate the four-modality Omni objective,
and does not clear the pretrained checkpoint for product use.

## Protocol and frozen artifacts

- Source branch commit for the audit pipeline: `1c2a8c13c173ec2b9eb48a76217b1c8b3936eaea`.
- Candidate: `clevr-vqa-windowed-readout-seed17-attempt-01`, selected at epoch 6
  using the previously frozen official CLEVR validation selector. Candidate
  selection was frozen before CoGenT-B annotations were accessed; no audit
  result was used for tuning or reselection.
- Base: `dandelin/vilt-b32-finetuned-vqa`, revision
  `d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08`, 117,588,537 parameters in its
  original form. The adapted frozen-encoder/readout configuration has
  111,616,540 parameters, of which 21,532 are trainable readout parameters.
  The existing FP32 weights are about 470 MB. This remains a research-only
  probe: VQAv2 training-data rights/provenance are not cleared for product use.
- Audit source: pinned CLEVR-CoGenT archive version
  `haDYjDZdal8NjYimkkMSeCanaOmhAc5y`, ETag
  `"a2125a3153354b3b09849befcdd79ad1-1572"`.
- Audit split: all 15,000 official `valB` images, exactly one deterministic
  seed-17 hash-ranked question per image. The frozen manifest has zero image
  overlap with train or development validation and zero normalized question
  overlap with development data. Manifest SHA-256:
  `4d4ca4c045f0cdb129011979422651304383c4c62a511be6b21ec7e563a5e5ac`.
- The 15,000 image files were downloaded from the pinned archive and checked
  against ZIP member names, lengths, CRCs, and per-image SHA-256. Feature
  extraction used the pinned ViLT model on CPU and preserved long questions by
  averaging overlapping 40-token windows (8-token overlap); 56 questions used
  more than one window. It produced three finite `[15000, 768]` matrices:
  real images, fixed gray images, and within-split shuffled images. Extraction
  took 10,628.3 seconds (2 h 57 m); no GPU or cloud compute was used.
- Feature matrix SHA-256 values: real
  `3038d2b3bc777ec15921d8e01eaf990644ca8ff5da28b62550a47aa6ec9671a7`, gray
  `bbc5ff371207b8eb0c0eedc7296d6709f6b9b92d180c63f677cb109a7daca9e3`, and
  shuffled
  `fd8742f19ac2f841ffc4745c5368fdb3c84424380de42745c8779f0226a8c08f`.
- Candidate-selection freeze SHA-256:
  `149ab2190c0a3c361357dc01ce6c44683958ed75d72022eb29312f2193252c6e`.
  Selected readout checkpoint SHA-256:
  `709fb32ba4af7669f3b6a6403a696b7d5d3625141d72da6f49f530236988bf7c`.
  Base weights SHA-256:
  `4d5f3409947b0369487ece7c5868f0040ceb67d25735dbb4ac5e99e03bab3a19`.
- Evaluation command: `python -m scripts.evaluate_clevr_cogent_b_audit`
  with the frozen audit manifest/metadata, attempt-02 CPU features and
  controls, frozen candidate-selection JSON, `best-readout.pt`, and
  `--bootstrap-replicates 10000`. The runner checked every artifact hash and
  candidate-selection freeze before writing results. GPU visibility was
  disabled.
- Evaluation result SHA-256:
  `c2563ef79d0a9a055394335198e789987a002d47aa6545dbeb37ab60ace615a2`.
  Real, gray, and shuffled per-example prediction SHA-256 values are recorded
  below with their paths.

## Results

Metrics are over supplied options. “Macro” gives equal weight to the six
answer kinds. ECE is reported descriptively and is not used to recalibrate.

| Input condition | N | Overall Accuracy | NLL | Brier | ECE | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real image + question | 15,000 | 57.90% | 0.7227 | 0.4732 | 0.0373 | 59.05% | 0.7740 | 0.4895 | 0.0434 |
| Fixed gray image + question | 15,000 | 55.10% | 0.7974 | 0.5154 | 0.0468 | 48.45% | 0.9164 | 0.5768 | 0.0982 |
| Same-split shuffled image + question | 15,000 | 51.81% | 0.8248 | 0.5346 | 0.0949 | 47.65% | 0.9536 | 0.5983 | 0.1352 |
| Uniform option probabilities | 15,000 | 39.94% | 0.9589 | 0.5977 | 0.0100 | 39.25% | 0.9918 | 0.6111 | 0.0123 |

Paired bootstrap uses 10,000 seed-17 percentile replicates, pairing by image
(one question per image). Deltas are real-image metrics minus the indicated
control; negative NLL/Brier is better.

| Paired comparison | Accuracy difference, 95% CI | NLL difference, 95% CI | Brier difference, 95% CI |
|---|---:|---:|---:|
| Real minus gray | +1.73 to +3.90 pp | -0.0816 to -0.0678 | -0.0472 to -0.0373 |
| Real minus shuffled | +5.26 to +6.91 pp | -0.1096 to -0.0946 | -0.0663 to -0.0568 |

| Answer kind | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Boolean | 6,143 | 48.25% | 0.7045 | 0.5111 | 0.0878 |
| Color | 1,330 | 51.73% | 1.1177 | 0.6093 | 0.0335 |
| Count | 3,613 | 74.59% | 0.5720 | 0.3327 | 0.0301 |
| Material | 1,226 | 62.89% | 0.6423 | 0.4515 | 0.0293 |
| Shape | 1,381 | 53.87% | 0.9574 | 0.5738 | 0.0483 |
| Size | 1,307 | 62.97% | 0.6502 | 0.4583 | 0.0316 |

| Outer question program | N | Accuracy | NLL |
|---|---:|---:|---:|
| Count | 3,613 | 74.59% | 0.5720 |
| Equal color | 709 | 38.50% | 0.7296 |
| Equal integer | 464 | 37.93% | 0.7561 |
| Equal material | 715 | 50.77% | 0.6987 |
| Equal shape | 693 | 41.27% | 0.7351 |
| Equal size | 750 | 51.73% | 0.6923 |
| Exist | 1,821 | 59.03% | 0.6609 |
| Greater than | 512 | 41.60% | 0.7273 |
| Less than | 479 | 39.67% | 0.7418 |
| Query color | 1,330 | 51.73% | 1.1177 |
| Query material | 1,226 | 62.89% | 0.6423 |
| Query shape | 1,381 | 53.87% | 0.9574 |
| Query size | 1,307 | 62.97% | 0.6502 |

The selected official-development checkpoint had 59.40% overall Accuracy,
0.7199 overall NLL, and 59.09% macro answer-kind Accuracy / 0.7688 macro NLL.
Those development results are separate from this held-out CoGenT-B audit.
The audit retains 57.90% overall and 59.05% macro Accuracy. The gap is small
in aggregate but meaningful for some kinds, especially Boolean and Shape;
question-program performance also varies substantially. This is evidence of
some transfer to the held-out color-composition split, not broad robustness.

## Interpretation and limits

The real-image condition beats both predeclared controls on Accuracy, NLL, and
Brier, with paired intervals excluding zero. Therefore the fitted readout is
using image information on this task; the result is not explained solely by
question-only priors. The 2.8-point overall Accuracy gain over blank images is
modest, and the six answer kinds range from 48.25% to 74.59%. In particular,
many Boolean and relational/count-comparison questions remain difficult.

This does not establish general multimodal fusion, video/audio/sensor ability,
95–98% quality, robustness across domains, or product deployment readiness.
CoGenT-B is an OOD validation split used once here, not a blind sealed product
test. The audit is closed to further tuning and repeated evaluation. No
ternary conversion, model promotion, or product-rights clearance follows.

## Reproducibility artifacts

All large/media/weight artifacts remain outside Git:

- Manifest and metadata: `C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-cogent-b-audit-seed17-attempt-01/`
- Verified media and acquisition receipt: `C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-cogent-b-audit-media-attempt-01/`
- CPU feature matrices and metadata: `C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-cogent-b-audit-features-seed17-attempt-02/`
- Evaluation outputs: `C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-cogent-b-audit-evaluation-seed17-attempt-01/`

Prediction file hashes:

- Real image: `592c7130e12e5f9d3a0d4dfac5bc97a9163a7e235cfce9f38d81c600e88c3050`
- Blank image: `667d0a32f38087ad834b603a2b1642cc1fd795e34ec003ea52fcd14244ef53c0`
- Shuffled image: `42495298951dc4b194a6469c963c161c2bf63a562d3f3ff91b426ffda8482506`

Runtime: Windows CPU, Python 3.11, PyTorch 2.6.0+cu124 (CPU execution),
Transformers 5.6.2, NumPy 2.4.6, Pillow 10.4.0. No GPU time, cloud compute,
or cloud cost was used for this audit. The separate existing Teacher GPU run
and its outputs were not modified.
