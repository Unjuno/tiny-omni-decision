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

## Additive evidence update — 2026-10-22

- **HoloAssist fixed-checkpoint AV replay:** The already-selected joint,
  audio-only, and video-only readouts were replayed once on the same 1,896
  labelled windows from 207 official-validation sessions. This validation is
  **not blind** because an earlier HoloAssist pilot used these sessions for
  checkpoint selection. Joint AV scored Accuracy/NLL/Brier/ECE-15
  `0.7711/0.5034/0.3291/0.0182`, audio-only
  `0.7600/0.5061/0.3325/0.0166`, and video-only
  `0.7194/0.5628/0.3795/0.0228`. Session-paired bootstrap intervals for
  audio-only minus joint include zero; video-only is lower on the paired
  comparison. Thus this replay does not establish that both inputs are
  necessary, does not approach the per-task quality target, and must not be
  used for tuning or selection. All official train and validation groups in
  this HoloAssist line have been exposed; do not run another HoloAssist
  selection experiment. The official test/sealed audit remained closed. Full
  hashes and protocol: [fixed-checkpoint replay](R2_HOLOASSIST_OFFICIAL_VALIDATION_FROZEN_REPLAY_20261022.md).
- **Perception Test AV candidate disposition:** The source, matched-input,
  scale, and label-support audits already exist in the Roadmap. Event category
  and visibility are strongly associated, and videos containing both states
  are not adjudicated counterfactual pairs. Keep its fixed readouts as weak,
  exploratory evidence; do not repeat the fit or call the task both-inputs-
  required. See [label-support audit](R2_PERCEPTION_AV_LABEL_STRATA_20261021.md).
- **AMI and AVQA admission status:** The separate AMI source-screen branch
  still has unresolved audio/video synchronization, visible-speaker truth, and
  content-overlap checks; it is not admitted for fitting. AVQA's `Both` relation
  is a useful task-design lead, but reviewed terms do not establish ML rights
  for annotations and YouTube-derived clips, so it is not admitted either. No
  new media or annotations were accessed for this update. See the [AMI source
  screen](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-ami-av-source-screen-20261021/docs/pretrained_reuse/R2_AMI_CROSSMODAL_SPEAKER_MATCH_SCREEN_20261021.md)
  and [AVQA rights recheck](R2_AVQA_RIGHTS_RECHECK_20261021.md).
- **Greatest Hits conditional-AV gate result:** The approved bidirectional
  exchange/gate design was already implemented and evaluated in the separate
  `codex/r2-av-postnorm-stability` workstream. Its seed-17 E0G1 run completed
  32,768 updates and selected step 24,576. Gate outputs collapsed below 0.01 by
  step 512. Although joint-view Accuracy was 5.07 points above E0G0, paired NLL
  uncertainty included zero and video-only Accuracy fell 3.83 points. In a
  fixed 8,875-question replay, the E0G1 full residual and query-only arm had
  identical Accuracy, with metric deltas around 1e-8. This is
  evidence against another gate fit on the same source and IDs, not evidence
  that the model family cannot fuse modalities. Greatest Hits remains a narrow
  exploratory task: native labels do not prove both-input necessity, and
  physical sync plus actor/site generalization remain unresolved. See the
  [matched-fit report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_GATED_POSTNORM_LONG_FIT_20261018.md)
  and [fixed-checkpoint diagnostic](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md).
- **Ego4D Social `Talking to me` source screen (2026-10-22):** This is a more
  task-aligned AV Decision candidate than the event-presence screens: given a
  synchronized segment and an annotated face track, decide whether that person
  is speaking to the camera wearer. The official task builds on speech-activity
  and face-track annotations and adds a directed-to-wearer label. Ego4D requires
  an accepted license agreement before annotations or media access; the public
  agreement endpoint was not readable in this screen, so permitted training,
  derivative-model and product use remain UNKNOWN. The public full-scale release
  is over 5 TB, although the official CLI supports benchmark-scoped subsets.
  Exact task group counts, current revision hashes, rights, and locally available
  assets have not been verified. Keep as REVIEW; do not acquire, fit or evaluate
  pending agreement review and a frozen group/split audit. Official sources:
  [AVD task](https://ego4d-data.org/docs/benchmarks/av-diarization/),
  [Social task](https://ego4d-data.org/docs/benchmarks/social/),
  [access terms](https://ego4d-data.org/docs/start-here/). A pinned public-code
  follow-up found that TTM expects AVD train/validation annotations and face
  tracks, while its older repository README still says benchmark data details
  are forthcoming. The AVD v1 clip counts are not verified TTM counts. Exact
  revisions, README hashes, and this discrepancy are captured in
  [`R2_EGO4D_SOCIAL_SOURCE_SCREEN_20261022.md`](R2_EGO4D_SOCIAL_SOURCE_SCREEN_20261022.md).
- **Ego4D license-draft follow-up (2026-10-22):** The official site's linked
  public draft is now readable. The reviewed agreement templates state that
  qualifying derived software/models may be used commercially, but access is
  granted only after the applicable non-transferable agreement is executed;
  the database itself cannot be redistributed or embedded in a product. This
  resolves the prior draft-availability uncertainty, not authorization to
  access the TTM data. Exact licensor/subset mapping, executed terms, counts,
  and usable split provenance remain open. No data was accessed. Keep AV7 in
  REVIEW. Details: [`license-draft review`](R2_EGO4D_SOCIAL_LICENSE_DRAFT_REVIEW_20261022.md).

## Additive evidence update — 2026-10-25

- **CLEVRER plus generated-tone timing v2:** The frozen four-arm, seed-17
  development run used 3,696 training rows across 773 scenes and 1,110
  task-specific validation rows across 240 scenes. The selected joint gated
  LSTM reached 83.51% Accuracy / 0.3534 NLL; matched audio-only reached 75.14%
  / 0.4577, with joint-minus-audio-only Accuracy difference +8.35 points
  (scene-paired 95% interval +5.89 to +10.83). Video-only and mean-pooled
  controls remained at chance. This is evidence of temporal AV interaction for
  this synthetic collision/tone task, not natural AV understanding: CLEVRER is
  silent, the audio is generated, and the validation media were already present
  in other CLEVRER tasks. The task-specific validation is exposed and must not
  be reused for fitting or checkpoint selection. The observed result narrows
  the R2-M question but does not close its both-input-required gate for natural
  multimodal quality or the broader matrix. Preserve the run as development
  evidence; see the complete [v2 protocol and result](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-clevrer-synthetic-av-coverage-20261022/docs/pretrained_reuse/R2_CLEVRER_SYNTHETIC_AV_COVERAGE_V2_RESULTS_20261022.md).

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
| TV1 (provisional, no quality fit) | Step text + observed cooking video; choose consistent / inconsistent | CaptainCook4D official annotations `a8a920a3293c4db27099a20ddbe3a3a9be1283e3` and downloader metadata `cf248fe2139fc6e37f9737141ec719948fbd1747` (Apache-2.0) | The 203 non-overlapping official-train recordings contain 2,956 labeled steps from only six participants. Five sampled train GoPro URLs and two sampled synced HoloLens PV URLs returned 404; one old local training recording decodes. Validation has only two participants. No model quality result. | Keep as a text+video plumbing candidate only. Require working authorized media and a participant-held-out development design with adequate independent groups before a quality fit; never use official test to repair group counts. Details: `docs/pretrained_reuse/R2_CAPTAINCOOK4D_MEDIA_GROUP_RECHECK_20261021.md`. |
| A1 | Audio-only, closed 10-way keyword choice | Speech Commands v0.02 `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da` (CC-BY-4.0) | One-time official-test comparison is now observed: frozen baseline 97.72% Accuracy, train-only temperature candidate 97.86%; speaker-bootstrap Accuracy difference includes zero. Calibration improved, but the task remains fixed-vocabulary keyword recognition. LibriSpeech four-choice transcript transfer is near chance. | Keep as a narrow audio reference, closed to tuning. Do not generalize to speech understanding; evaluate any future frozen candidate only on a newly sourced, independently grouped task. |
| V1 | Video + question + choices, with static and temporal subsets reported separately | CLEVRER `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef` (CC0-1.0) | Fresh selected VideoPrism + MiniLM readout and legacy v0 were near binary chance on a one-time 200-scene holdout; paired intervals include zero. Separate long-budget descriptive results show severe late NLL/ECE degradation and no additional unique Video questions. | Use only as a synthetic temporal reasoning challenge. Group every question and choice by video. Report question types separately. The already observed holdout and validations are closed to further selection; admit new authorized, scene-disjoint Video supervision before another quality fit. |
| AV1 (exploratory; closed to further selection) | Time-localized audio + synchronized video; decide whether the sound source is visible | Google DeepMind Perception Test `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; exploratory materials CC-BY-4.0 per pinned official repo | Matched readouts over 256 video-disjoint train and 256 validation events reached 100% train Accuracy and only 56–65% validation balanced Accuracy. Full-split visible priors differ (74.4% train vs 91.3% validation), and event category strongly predicts visibility. The selected validation has no adjudicated within-video counterfactual pairs; category/label support is weak for proving both-input necessity. | Retain only as negative exploratory evidence. Do not refit or tune on these cohorts. A new Perception Test experiment would need an independently selected task/cohort whose labels establish conditional multi-input necessity; prefer a different admitted source. See the [source screen](R2_PERCEPTION_TEST_SOURCE_SCREEN_20261021.md) and [label-support audit](R2_PERCEPTION_AV_LABEL_STRATA_20261021.md). |
| AV2 (exploratory, not admitted) | Given an 8-frame video interval, choose its synchronized audio window from two same-video, same-label event windows | Perception Test `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; annotation-derived target, no official test split | One fixed seed-17 development corpus has 190 train and 313 validation video groups. Pooled joint Accuracy was 47.60% versus audio-only 50.80%; paired group interval included zero. A 19,009-parameter conditional gate overfit by update 32 and its minimum-NLL selector chose step 0. The readout used pooled features, so it is not a temporal architecture test; validation labels/source were already inspected. | Keep as a negative development diagnostic, not a product task or fusion pass. Do not continue tuning on these IDs. A future both-required task needs independently verified labels, sequence-level inputs and a new predeclared group split. Full metrics and hashes: `docs/pretrained_reuse/R2_PERCEPTION_AV_TEMPORAL_MATCH_20261021.md`. |
| AV3 (exploratory replay; closed to selection) | HoloAssist synchronized audio/video + task category; predict the annotated action-state window label | HoloAssist official train/validation release; exact source/cache identities and hashes are in the linked replay report | Joint AV reached 77.11% Accuracy on 1,896 labelled windows / 207 validation sessions; audio-only reached 76.00%, video-only 71.94%. Audio-only versus joint paired session intervals include zero. The validation was previously exposed by an earlier pilot; all groups in this source line are now exposed. | Historical transfer evidence only. It does not prove both modalities are required and cannot qualify broad audio, video, or AV quality. Do not tune, select, or rerun on these groups. Official test/sealed audit remains closed. See `R2_HOLOASSIST_OFFICIAL_VALIDATION_FROZEN_REPLAY_20261022.md`. |
| AV4 (review-only) | AMI synchronized conversation audio/video; decide which candidate face is the active speaker for a time-localized utterance | AMI Meeting Corpus, using complete parent meetings and official SA/SB grouping | A source-screen subset has 214 preliminary train and 111 development windows; 213/214 train joins pass, with one quarantined. Metadata supports SA/SB participant separation, but exact A/V synchronization, visible-speaker truth, and content-overlap proof remain incomplete. | Do not fit or call this a joint task yet. Admit only after synchronization and visible-speaker labels are independently verified, media/content overlap is audited, and grouped task construction is frozen. Details are on the [AMI source-screen branch](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-ami-av-source-screen-20261021/docs/pretrained_reuse/R2_AMI_CROSSMODAL_SPEAKER_MATCH_SCREEN_20261021.md). |
| AV5 (rights-blocked) | AVQA `Both` questions: answer a multiple-choice question whose annotation says audio and visual information are related | AVQA annotations over VGG-Sound/YouTube clips | The official task description contains a useful `View` / `Sound` / `Both` relation, but reviewed annotation terms and original clip rights do not establish machine-learning use rights. | Do not fetch annotations/media, train, or evaluate until the relevant rights are explicit. Keep only as task-design reference; see `R2_AVQA_RIGHTS_RECHECK_20261021.md`. |
| AV6 (exploratory; gate comparison complete) | Audio + video event; choose released material/action/reaction fields | Greatest Hits official release, CC BY 4.0; development partition grouped by recording date | The matched seed-17 E0G1 gate run completed 32,768 updates (best step 24,576); gate values collapsed below 0.01 by step 512. The joint-view Accuracy gain over E0G0 was +5.07 points, but paired NLL uncertainty included zero and video-only Accuracy declined 3.83 points. Full residual and query-only arms had identical Accuracy, with metric deltas around 1e-8, in a fixed 8,875-question replay. | Do not repeat this gate run or use its labels as proof of both-input-required truth. Keep as scoped development evidence. A follow-up requires a fresh admitted task with defensible synchronization, group independence and truth-bearing modality controls. Reports: `R2_AV_GATED_POSTNORM_LONG_FIT_20261018.md` and `R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md` on branch `codex/r2-av-postnorm-stability`. |
| AV7 (review-only; access/rights gate open) | Given an audio-video conversation segment and a supplied visible-person track, choose whether that person is talking to the camera wearer | Ego4D Social `Talking to me`, built on AV Diarization annotations; public license agreement required before annotation or media access | The official task combines visible-face tracks, active-speaker annotations and a directed-to-wearer label. The Ego4D release covers 926 camera wearers, but usable Social-task groups/counts, exact source revision, terms, and local media availability remain unverified. Full-scale data is >5 TB; the official CLI supports benchmark-scoped subsets. | Strong candidate for a supplied-track AV Decision task, not admitted. Do not fetch or fit until the actual license agreement is reviewed and accepted by the user/rights holder, then verify group-video/participant splits, media identity, synchronization and availability. Compare matched audio-only, video-only and joint controls; the task label alone does not prove both streams are indispensable. Official sources: [AVD](https://ego4d-data.org/docs/benchmarks/av-diarization/), [Social](https://ego4d-data.org/docs/benchmarks/social/), [access](https://ego4d-data.org/docs/start-here/). |
| AV8 (task-design lead; rights/split gate open) | Text query + audiovisual clip gallery; retrieve the target clip, focusing on queries whose answer needs both audio and video | FLARE `AnonymousFLARE/FLARE@ed89d868647a3658fe6989ba06b7a7117c2624d9` (dataset card declares CC-BY-4.0; 399 source videos from Video-MME) | FLARE reports 53,580 hard-bimodal queries, but its release includes 13 ZIP volumes of segmented Video-MME media. Video-MME's official terms restrict use to academic research, prohibit commercial use, and restrict copying/distribution. The inspected viewer shows one `test` split, with no established group-disjoint train/validation/audit split. A few public preview caption rows appeared in screening and were not retained. | Not admitted for fit, quality evaluation, or packaging pending written rights covering the original videos and a new authorized, video-grouped protocol. Retrieval ranks would also need a preregistered conversion into fixed supplied-option Decision evaluation with calibration metrics. See [FLARE source screen](R2_FLARE_SOURCE_SCREEN_20261022.md). |
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
- Next work remains a source/task gate, not another architecture fit. The
  Perception Test visibility readout and HoloAssist line are closed to further
  selection; AVQA is rights-blocked, and AMI still lacks verified synchronization
  and visible-speaker truth. Ego4D Social `Talking to me` and FLARE are
  promising task-design leads, but their applicable media rights and usable
  group-disjoint development sets are unresolved. Continue only after rights,
  task truth, and fresh group-disjoint development records are established.
  Freeze the task and selection design before quality-selected training. Do not
  open sealed audits to decide among candidates.
