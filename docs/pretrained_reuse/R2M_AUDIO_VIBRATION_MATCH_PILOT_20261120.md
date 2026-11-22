# R2-M audio + vibration matching pilot (2026-11-20)

## Outcome

The bounded pilot completed with the frozen train/calibration/validation
participants and never opened sealed-audit signals. A frozen EfficientAT audio
representation plus a 178,563-parameter time-aware gated readout did not learn
same-trial correspondence: the selected validation checkpoint remained at
chance-level binary performance and below chance on the paired two-candidate
choice metric. A fixed energy-correlation baseline was also near chance. This
does not establish a ceiling for the source, the architecture family, or the
overall multimodal Decision objective. It does mean this matching formulation
and feature path do not currently justify product use.

This is a synthetic correspondence task: given audio and vibration, decide
whether the streams came from the same activity trial. Negatives are another
trial of the same activity with similar duration; they are deliberately harder
than random negatives. It is not an end-user action-selection benchmark and
does not measure the 95–98% product target.

## Source, split, and pairing

- Source: Zenodo `10.5281/zenodo.7937591`, CC BY 4.0; the previously verified
  local SHA-256 receipt was rechecked for all 24 active archives.
- Source member: `multisensor-1` only.
- Frozen split v2 SHA-256:
  `0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0`.
- Participant groups: train 7, calibration 2, validation 2, sealed audit 2;
  participant 0 remains quarantined and participant 999 is excluded.
- The indexer read identifiers to select rows; audio/vibration arrays were
  dereferenced only for train/calibration/validation participants. Sealed
  participants 1 and 16 and quarantined participant 0 were skipped before any
  signal-array access.
- Synchronization was required within 0.5%; same-activity negative duration
  tolerance was fixed at 10%. Cross-participant negatives were preferred, then
  duration difference and deterministic seeded tie-breaks selected the trial.
  Negative reuse was minimized.

| Split | Allowed trials | Paired questions | Unpaired anchors | Cross-participant negatives | Within-participant negatives |
|---|---:|---:|---:|---:|---:|
| Train | 5,464 | 5,448 | 16 | 5,386 | 62 |
| Calibration | 779 | 759 | 20 | 0 | 759 |
| Validation | 775 | 761 | 14 | 395 | 366 |

Calibration has only within-participant negatives because its two participants
share no activity. Validation results are therefore reported by negative
relation. Validation contains two participants only; its rows are not
independent participant-level replications.

Pair-index internal digest:
`d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276`; JSON
file SHA-256 `33664b32205626fe1ec2102961e82163d2f55586f6c0567c9f1c80c0f4d3e7bf`.
Allowed trial-metadata file SHA-256:
`faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0`. Raw
media and labels were not copied into Git.

## Frozen features and model

- Audio encoder: EfficientAT DyMN-0.4 AudioSet, upstream revision
  `a425fdce92572e602a1d5634799bd9f1f2efa806`, checkpoint SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`;
  1,961,239 parameters, all frozen.
- Preprocessing: int16 audio to float, 16 kHz to 32 kHz sinc resampling,
  official eval-mode AugmentMelSTFT, final convolutional feature map averaged
  over frequency while preserving time. Each trial retains variable-length
  `[time,384]` tokens.
- Vibration: three axes resampled deterministically to 128 time tokens; no
  learned sensor encoder. Raw waveforms were not written to the feature cache.
- Cache: 6,971 unique trial assets used as anchors or negative donors; 5,449
  train, 760 calibration, and 762 validation. Cache-index SHA-256
  `61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498`.
- Candidate model: `TimeAwareGatedFusion(audio=384,sensor=3,query=2,width=96,
  heads=4)` plus a shared scalar option scorer; 178,563 trainable parameters.
  The only options are “not same trial” and “same trial.”
- Per anchor, one positive and one matched negative binary example are used.
  This is 10,896 train examples per epoch, 1,518 calibration examples, and
  1,522 validation examples.
- Seed 17; AdamW, LR `1e-3`, weight decay `0.01`; cosine scheduler over 80
  epochs, no warmup; batch 16; cross-entropy; no early stop. All 80 epochs ran.
  The checkpoint selector was minimum calibration NLL. Best checkpoint epoch
  16 was kept separately from final epoch 80.
- Fixed baseline: zero-lag Pearson correlation between interpolated temporal
  norms of EfficientAT frame embeddings and vibration-vector magnitude. It was
  not fitted or calibrated on validation.

## Learning curve

Rolling training CE is the per-minibatch mean and is **not** a fixed-example
generalization-gap measurement. Calibration rows are used for checkpoint
selection. Validation is reported once using the selected checkpoint after the
full fixed budget.

| Epoch | Updates | Rolling train CE | Calibration Accuracy | Calibration NLL | Calibration Brier | Calibration ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 681 | 0.699554 | 0.4993 | 0.6931471 | 0.4999999 | 0.000665 |
| 5 | 3,405 | 0.693152 | 0.5000 | 0.6931472 | 0.5000000 | 0.000021 |
| 10 | 6,810 | 0.693148 | 0.5000 | 0.6931472 | 0.5000000 | 0.000002 |
| **16 (selected)** | 10,896 | 0.694406 | 0.5000 | **0.6931448** | 0.4999976 | 0.000081 |
| 20 | 13,620 | 0.693180 | 0.5000 | 0.6931472 | 0.5000000 | 0.000001 |
| 40 | 27,240 | 0.693164 | 0.5000 | 0.6931472 | 0.5000000 | 0.000060 |
| 60 | 40,860 | 0.693158 | 0.5000 | 0.6931472 | 0.5000000 | 0.000033 |
| 80 (final) | 54,480 | 0.693147 | 0.5000 | 0.6931472 | 0.5000000 | 0.000006 |

The calibration selector's best is only a tiny NLL change from uniform chance;
Accuracy never improves. The fixed budget reached its cosine endpoint at epoch
80. There is no evidence of useful task learning in this run, and no stable
train/validation separation was measured.

## Validation results

Candidate-level binary option metrics (1,522 decisions; chance Accuracy 50%,
chance NLL `0.693147`, chance Brier `0.5`):

| System | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Gated, selected epoch 16 | 0.4993 | 0.693149 | 0.500001 | 0.000744 | 0.500087 |
| Fixed temporal-correlation baseline | 0.4875 | 0.695052 | 0.501871 | 0.033841 | 0.521357 |

The more direct task is to select the matching vibration from the paired
positive/negative candidates (761 questions). The probabilities for “same
trial” were converted to one score per candidate and normalized within each
question; this was a post-hoc calculation over saved predictions, not another
model forward or checkpoint-selection step.

| System | Pair choice Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Gated, selected epoch 16 | 0.4534 | 0.693150 | 0.500003 | 0.046675 |
| Fixed temporal-correlation baseline | 0.5059 | 0.696561 | 0.503355 | 0.023467 |

| Pair-choice stratum | Gated Accuracy / NLL | Fixed baseline Accuracy / NLL | Questions |
|---|---:|---:|---:|
| Cross-participant | 0.4861 / 0.693156 | 0.4962 / 0.698798 | 395 |
| Within-participant | 0.4180 / 0.693144 | 0.5164 / 0.694146 | 366 |

Both systems are near chance, and the gated model is below chance on the
paired selection metric. No uncertainty interval is presented: validation has
only two participant clusters, so a row-level interval would overstate
independence. This is development validation, not blind audit evidence.

### Fixed-weight controls

| Validation condition | Binary Accuracy | NLL | Brier | Paired candidate Accuracy |
|---|---:|---:|---:|---:|
| Normal selected model | 0.4993 | 0.693149 | 0.500001 | 0.4534 |
| Audio absent | 0.5000 | 0.693165 | 0.500018 | 0.5112 |
| Sensor absent | 0.5000 | 0.694112 | 0.500963 | 0.0000 |
| Sensor timestamps shifted +250 ms | 0.4993 | 0.693149 | 0.500001 | 0.4376 |
| Positive audio paired with same-activity wrong-trial sensor, labeled mismatch | 0.0407 | 0.693313 | 0.500166 | n/a |

Outputs are close to uniform and mostly insensitive to the controls. The
4.1% accuracy on the all-mismatch control reflects a near-tie prediction bias,
not demonstrated sensor reliance. These controls do not pass the multimodal
necessity gate.

## Hardware, runtime, and artifacts

- Device: local NVIDIA RTX 3080 Laptop GPU, 16 GiB; GPU training only. The
  separate Orbitune process was not stopped manually.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, torchaudio 2.6.0+cu124, CUDA
  12.4, h5py 3.16.0. EfficientAT source and checkpoint were already present;
  no packages were upgraded.
- Feature extraction: CPU FP32, one thread, 6,971 features in 650.3 seconds;
  186,252,583 bytes across feature bundles and receipts.
- Pilot wall time: 1,017.3 seconds (about 17 minutes), including training,
  calibration checks, checkpoint save/reload, final validation and controls.
  Final validation and controls took about 5.7 seconds after epoch 80.
- Peak PyTorch CUDA allocation: 103,123,968 bytes (about 98.3 MiB). Cloud cost:
  zero. The measured GPU memory is model allocation; it excludes driver/context
  overhead.
- Training-start artifacts were created at 2026-11-20 22:41:56 JST; final
  metrics at 22:58:55 JST. Recorded wall duration is authoritative.
- Best checkpoint SHA-256:
  `5de528a7f0a19b1c17cf487151edbd68c0b14defc4b58348f4797189b1d01542`.
- Final checkpoint SHA-256:
  `16b28f44c14d3adf885c3b926ace060898c347ab06f59b75f504d307f4aa015f`.
- Training history SHA-256:
  `e8f785926d00574a74c573a81b252b02039d4799403ff1b248786b2346ecb32c`.
- Effective config SHA-256:
  `daea58fc8b22eac82aa3ad280811de47d6aa736f62176b4a4250b4bbc74f86`.
- Validation predictions SHA-256:
  `fd553a1c295f55bd487c6a9f349b8a24027fd8d0194a312925429dcc04c16981`.
- Paired-choice post-hoc metrics SHA-256:
  `aac079eb65467590d8e6e82259fd8384ba8f0cd86506b1deb4b59743215d0931`.

The run used Python 3.11.9 and the existing local package directories; nothing
was installed or upgraded. The following are the exact commands (output paths
are already occupied and must not be reused for a new attempt):

```powershell
$env:PYTHONPATH='C:\CodexWorktrees\pretrained-reuse-next-20261115\src;C:\CodexArtifacts\pretrained-reuse-r1\python-deps;D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\python-packages;C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\source'
$python='C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
& $python scripts\build_r2m_audio_vibration_pair_index.py --artifact-root 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1' --output-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01'
& $python scripts\extract_r2m_frozen_features.py --artifact-root 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1' --pair-index-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01' --efficientat-source 'C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\source' --checkpoint 'C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\dymn04_as.pt' --output-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-frozen-feature-cache-v1-attempt-02'
& $python scripts\train_r2m_gated_decision_pilot.py --cache-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-frozen-feature-cache-v1-attempt-02' --pair-index 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01\pair-index.json' --output-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-gated-decision-pilot-v1-attempt-01' --device cuda --epochs 80 --batch-size 16 --learning-rate 0.001
& $python scripts\summarize_r2m_pair_predictions.py --pair-index 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01\pair-index.json' --model-predictions 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-gated-decision-pilot-v1-attempt-01\validation-predictions.jsonl' --baseline-predictions 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-gated-decision-pilot-v1-attempt-01\validation-temporal-correlation-baseline.jsonl' --output-dir 'D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-paired-choice-posthoc-v1-attempt-02'
```

All model weights, media, feature tensors, histories, and predictions are
outside Git under
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\`.
The pilot output is `r2m-gated-decision-pilot-v1-attempt-01`; the full pair
index is `r2m-full-pair-index-v1-attempt-01`; the feature cache is
`r2m-frozen-feature-cache-v1-attempt-02`. Teacher v1, prior v2 artifacts,
frozen source splits, and checkpoints were not overwritten. Ternary
quantization, recovery training, and sealed audit were not performed.

## Integrity and provenance notes

The training run used base repository commit `31022cc1ec0c9d42eeb18e76dd1a1df67008db06`
plus uncommitted pilot scripts. The launch-time training-script file hash was
not captured before execution. The stored `source_commit` field identifies the
base commit, not a complete source snapshot; exact launch-tree hash is UNKNOWN.
The current training script hash was recorded after adding paired-choice
reporting helpers, so it is not substituted for the missing launch hash. Those
post-run edits affect derived metrics only, not the training loop or saved
weights. This provenance gap is retained rather than guessed.

The earlier feature smoke recorded hash
`34528184e6cac158fdfa69b7beaa5ca03d7a1c3fd830a4417da848c5e4e08695`, while
the reproducible current extractor produced
`e662c72ef734c18ca22ee65cc0431743786e33651ea22e788b1ae9591f64349b` twice.
The old run did not save its tensor or exact runtime/thread settings, so the
difference is UNKNOWN. The current feature cache is internally hash-verified;
equivalence to the earlier smoke is not claimed.

The first full-extractor attempt stopped before dereferencing any source signal
because the trial metadata lacked an explicit `sample_id` field. It created an
empty attempt directory and no cache. Attempt 02 added the deterministic ID
derivation and completed. Both attempt directories remain intact.

## Decision and next step

**R2-M same-trial matching: NOT SUPPORTED by this candidate.** The complete
data budget and 80-epoch run were consumed, but the gated model did not beat
chance and did not show useful dependence on both inputs. Do not escalate this
readout into the product Teacher and do not use the sealed audit to diagnose
it. The fixed correlation baseline is also inadequate.

Before another architecture run, inspect the source's sensor-unit/axis
semantics and audio-vibration capture synchronization, and determine whether
same-trial identity is observable after the chosen feature transforms. If the
source supports an actual action Decision (for example activity identification
over supplied choices), define that task with split-safe labels and both-input
necessity controls; otherwise close this source task as non-informative. Any
new experiment must use a separate artifact directory and preserve v1/v2 and
this baseline.
