# R0 task battery and device proposal (draft)

Status: **decision draft, not a frozen acceptance contract**. This document
organizes existing source and run evidence; it does not authorize opening any
sealed split, promote a model, or start a new GPU experiment.

## Additive evidence update — 2026-10-21

The following completed comparisons postdate the initial draft. Their evaluated
splits are now observed reference evidence and must not be reused for tuning or
checkpoint selection.

- **A1 Speech Commands:** The frozen 10-keyword readout was evaluated once on
  4,074 retained official-test rows from 239 speakers. Uncalibrated metrics
  were Accuracy/NLL/Brier/ECE-15 `0.9772/0.2684/0.0909/0.1692`; a candidate
  with scalar temperature fitted on separate train-only speakers reached
  `0.9786/0.0946/0.0372/0.0237`. The paired speaker-bootstrap Accuracy
  difference includes zero. This supports only this fixed-vocabulary keyword
  task; it is not open-vocabulary speech or general audio understanding. Keep
  both outputs as historical reference and do not make later decisions from
  this test split. Full protocol and hashes are on the [evaluation branch](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-clevrer-fresh-reasoning-20261021/docs/pretrained_reuse/R2_TEXT_FREE_AUDIO_OFFICIAL_TEST_PROTOCOL_20261021.md).
- **V1 CLEVRER reasoning:** A fresh VideoPrism + MiniLM readout was selected
  on 200 development scenes, then compared once on a separate 200-scene,
  2,843-choice holdout. Macro-by-question-type Accuracy/NLL/Brier/ECE-15 were
  `0.4966/0.6934/0.5002/0.0145`, versus legacy v0
  `0.4933/0.6953/0.5022/0.0313`; every paired scene-bootstrap interval
  includes zero. Both remain near binary chance. This holdout is now observed
  and closed to further selection. Full split, hash, and software-recovery
  record: [fresh CLEVRER execution](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-clevrer-fresh-reasoning-20261021/docs/pretrained_reuse/CLEVRER_FRESH_REASONING_V1_STATUS_20261021.md).
- **Path A long-budget coverage comparison:** The matched 32,768-update
  fresh-coverage arm selected step 8,192, just like the repeat control. The
  selected fresh checkpoint was slightly worse on all four macro metrics. At
  the fixed final step, Fresh had higher Accuracy and lower NLL/Brier/ECE than
  Repeat; both arms nevertheless had severe NLL/ECE degradation relative to
  their own selected checkpoint. Fresh coverage added no unique Video
  questions. Treat this as evidence that the tested Video training pool was
  narrow and overfit, not as a general architecture ceiling or authorization
  to tune on the reused validation. Details: [matched long-budget
  results](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-native-coverage-long-20261021/docs/pretrained_reuse/R2_NATIVE_COVERAGE_LONG_RESULTS_20261021.md).
- **CI for the reconciled roadmap evidence:** GitHub Actions runs `39589674150`
  and `39589667615` both completed successfully on `ae5ac12ace1446ea6d0cdfd747482d82099eda2f`;
  install, lint, tests, and manifest validation all passed. This verifies the
  PR head, not the scientific quality gates below.

## Product target and measurement scope

- Product objective: a local, small Decision system accepting typed JSON inputs
  and returning a calibrated probability distribution over caller-supplied
  option IDs. Generation quality is out of scope.
- Engineering host currently available: Windows PC with an RTX 3080 Laptop
  GPU (16 GiB). This is a measured reference host, not yet the selected
  end-user/deployment device. Report CPU, GPU, package versions, cold/warm
  latency and peak memory separately for each runtime profile.
- Provisional engineering budget already authorized: 100–200M resident
  pre-quantization parameters for the complete runtime, and <=200 MB for the
  complete post-quantization package. Keep parameter count and package bytes as
  separate gates. These values are not release guarantees.
- User's quality objective is >=95% Accuracy per required task, with 98% as an
  aspiration. Never average unrelated tasks or modalities to pass. NLL, Brier,
  ECE-15, confidence intervals, coverage and per-class results remain required
  reported measures. Numeric calibration/abstention thresholds and latency
  limits are not yet approved.
- Compute is local with $0 cloud spend. A GPU process may only be started via
  the existing cooperative HotUGPU reservation path after data and runtime
  preflight; it must not signal or stop another workload. No GPU process was
  started for this draft.

## Proposed task battery

| ID | Inputs and decision | Candidate pinned source | Evidence and current limitation | Proposed gate use |
|---|---|---|---|---|
| T1 | Text state + question + 3–5 supplied choices; return option probabilities | Typed Decisions Synth `n4ze3m/typed-decisions-synth@5ece89a225b23c4cd5c4bab5735a0819d61dd7d5` (MIT); Open-Jev only after access is restored and source groups are verified | Current 58.11% adapted Cross-Encoder result is synthetic. Open-Jev access returned HTTP 401 in the recorded audit. | Keep as controlled text/pipeline diagnostic; it cannot alone qualify factual/general text understanding. Require an independent state/template/content-held-out set before product evaluation. |
| I1 | Image + fixed taxonomy question; choose among supplied attributes/count options | CLEVR-4 `sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73` (CC-BY-4.0) | Best reported full-coverage result is synthetic; macro Accuracy 88.06% on a one-time existing validation subset, with count at 61.19%. Other image readouts and spatial controls also leave counting weak. | Keep as a reproducible visual attribute/count challenge. Report each taxonomy independently; no natural-image claim. Existing validation is no longer blind. |
| A1 | Audio-only, closed 10-way keyword choice | Speech Commands v0.02 `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da` (CC-BY-4.0) | One-time official-test comparison is now observed: frozen baseline 97.72% Accuracy, train-only temperature candidate 97.86%; speaker-bootstrap Accuracy difference includes zero. Calibration improved, but the task remains fixed-vocabulary keyword recognition. LibriSpeech four-choice transcript transfer is near chance. | Keep as a narrow audio reference, closed to tuning. Do not generalize to speech understanding; evaluate any future frozen candidate only on a newly sourced, independently grouped task. |
| V1 | Video + question + choices, with static and temporal subsets reported separately | CLEVRER `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0) | Fresh selected VideoPrism + MiniLM readout and legacy v0 were near binary chance on a one-time 200-scene holdout; paired intervals include zero. Separate long-budget descriptive results show severe late NLL/ECE degradation and no additional unique Video questions. | Use only as a synthetic temporal reasoning challenge. Group every question and choice by video. Report question types separately. The already observed holdout and validations are closed to further selection; admit new authorized, scene-disjoint Video supervision before another quality fit. |
| AV1 | Time-localized audio + synchronized video; decide whether the sound source is visible | Google DeepMind Perception Test `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; exploratory materials CC-BY-4.0 per pinned official repo | 512 selected media pairs and frozen features are integrity-checked. The 128/class pilot's tiny heads reached 100% train and only 56–65% validation balanced Accuracy. Official full splits are strongly imbalanced: visible share 74.4% train vs 91.3% validation. A separate fresh independent-group task result has not qualified fusion. | Candidate for M-2 only after timing/label audit and a predeclared group-held-out cohort. Compare audio-only, video-only and joint on identical video groups; include missing, shuffled, stale and timing-offset controls. Require a measured joint improvement and evidence both inputs are useful before calling this complementary AV quality. |
| S1 (optional) | Time-stamped sensor window + question/options; explicit units, missingness and observation-end time | VIDIMU only if its acquisition and split provenance remain verified; do not substitute UCI HAR while MOMENT pretraining overlap is unknown | Current pilot has only 39 development trials / 3 participants and no independent high-powered quality evidence. Sensor-label mapping and group coverage are insufficient for a release task. | Keep optional and research-only until >=200 independent subjects/scenes are available or a justified low-power protocol is approved. Never pool it with the four required modalities. |

## Common acceptance protocol proposal

1. Before any quality-selected fit, freeze exact task wording, label mapping,
   source revision, group split, normalized-content/media hashes, option
   construction, sample order, stopping/selection rule and calibration split.
2. Use train groups for fitting, distinct train-only groups for calibration,
   validation groups for one predeclared selector, and a new sealed audit only
   once the model and policy are frozen. Existing inspected validation sets are
   development evidence and remain labeled as such.
3. Report each task and source independently: Accuracy, balanced/macro Accuracy
   when class imbalance exists, NLL, multiclass Brier, 15-bin ECE, mean/max
   confidence, per-class confusion, example count, independent group count, and
   a 10,000-resample group bootstrap with seed 17. Fewer than 200 independent
   groups is low-power and cannot close the product quality gate.
4. The research target for each eventual required task is >=95% Accuracy; 98%
   is stretch. No metric thresholds for calibration/abstention are set here;
   no task is qualified merely because it meets Accuracy on a previously used
   validation set.
5. A joint task is admitted only if labels and controls show why the decision
   needs multiple inputs. Synchronized media, replacement sensitivity, or
   separate unimodal accuracy is insufficient. Compare against matched
   best-unimodal systems and use identical examples/options.
6. Record total resident parameters before quantization; after quality is
   established, record actual packed bytes including weights, scales,
   exceptions, tokenizer/preprocessing, metadata and runtime assets. No
   quantization decision is part of this draft.

## Open decisions and next actions

- User approval is still needed before calling the Windows/RTX 3080 host the
  actual end-user target. For now it is only the engineering benchmark host.
- The named tasks above are candidate task slices, not yet a final required
  product battery. Open-Jev access, an unobserved text set, new blind image
  evidence, a useful video task, AV complementarity, and optional sensor
  coverage remain unresolved.
- The fixed-seed audio calibration candidate has now received its one-time
  official-test comparison. Keep the measured result in its narrow scope; do
  not retune against that test or describe it as broad audio quality.
- Next work remains a source/task gate, not another architecture fit: audit the
  AV1 cohort's event truth and video-grouped evaluation support, and select one
  untouched, representative task per required modality before any product
  claim or quality-selected training. Do not open sealed audits to decide among
  candidates.
