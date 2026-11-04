# Completed native audiovisual command comparison

## Result and decision

The frozen three-arm comparison **actually completed**: audio-only, video-only,
and joint each ran24 epochs/768 optimizer updates on the same4,080 training
assets. Complete native feature banks passed source, identity, parameter,
hash and reload checks before learning. All selected/final checkpoint replays,
saved logits/probabilities/JSON, optimizer records, sampler, train-only
normalization, calibration-only temperatures and frozen controls passed.

Audio-only reaches **97.94% macro slot Accuracy**; joint reaches **97.09%**;
video-only reaches **30.15%**. Joint-minus-audio Accuracy is **−0.844 percentage
points**, paired speaker-group95% interval **[−1.498, −0.260]**. Its NLL increase
is0.033317[0.021074,0.050759], Brier increase0.015023[0.008496,0.023279].
Do not adopt this pooled audiovisual fusion or call aggregate97% an Omni success.

The task is **six closed-vocabulary speech-command slots**, with all51 options
retained, including25 letters. Each slot has889 validation utterances. Macro
Accuracy averages six slots; it is different from all-six-slots-correct utterance
Accuracy. Audio's weakest slot(letter) is93.48% and all-six-correct is88.41%; joint's
letter Accuracy is90.66% and all-six-correct83.91%. Thus even this narrow task has
not reached95% on every slot or complete-command decision. Text/image/general
video/condition-following/sensor/Omni quality is not established by these results.

## Fixed protocol and integrity

- Source: publisher's CC-BY4.0 Lombard Grid corpus, snapshot/archive SHA pins in
  [source manifest](../../manifests/pretrained_reuse/lombard-grid-source-20261014.json).
- Fixed source/speaker/complete intended-and-actual phrase groups; train4,080,
  calibration398, validation889. Validation has9 speakers/groups. No held-out IDs
  were removed; four invalid training videos and19 unresolved labels remain
  separately recorded. No labels, transcripts, filenames or alignments are neural
  features. Pretraining overlap remains UNKNOWN; this is development validation,
  not a new blind audit. No old Teacher/sealed data was opened or changed.
- [Frozen protocol](../../configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json)
  SHA`047e98485298c5ff26900e20335aa49a1aa8cbf0581772482edc2dad3c19afe4`.
- Native frozen Whisper Encoder8,208,384 parameters and VideoPrism114,365,184
  parameters; complete physical speech token mean384D and native16-frame
  spatial/time pooled visual mean768D. Actual-PTS sampling,288 crop,FP32 unchanged.
  The failed numerical Torch port was not used or promoted.
- Head1154→64→51, GELU,77,235 trainable parameters. Same fresh seed17
  initialization, same independent seed17 sample orders, batch128, AdamW,
  fixed LR0.001/weight decay0.001,24 epochs. Encoders were not loaded in training.
- Earliest minimum unrounded **raw validation macro NLL** selects best. Final24
  saved separately. Per-slot17 geometric temperatures0.25..4 fit only calibration.
  Selected audio/joint happen to be24; video selects3. No extra seeds, epochs,
  architecture changes or observed-score configuration selection were performed.
- Each arm consumed97,920 assets/587,520 slot decisions, with4,080 unique assets,
  24,480 unique slot decisions and93,840 repeated asset consumptions. Selected
  audio/joint consumed the full97,920; selected video consumed12,240 assets/
  73,440 slot decisions through epoch3. All arms still completed24 epochs.
- Exact split, input-feature, config, sampling, initial/best/final tensor and
  report hashes are retained in the [adjacent result JSON](R2_LOMBARD_GRID_RESULTS_20261014.json).

## Measured tables and curves

The following tables are mechanically copied from the audited saved outputs.
Best calibration does not alter argmax Accuracy. Raw/final and all24 learning
points remain available in JSON and [curve CSV](R2_LOMBARD_GRID_CURVES_20261014.csv).

### Selected calibrated checkpoints

| Arm | Epoch | Macro Accuracy | NLL | Brier | ECE | Mean confidence | Min slot | All-six correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audio | 24 | 97.94% | 0.056971 | 0.029461 | 0.010362 | 0.981321 | 93.48% | 88.41% |
| video | 3 | 30.15% | 1.740170 | 0.767596 | 0.039470 | 0.299020 | 6.86% | 0.00% |
| joint | 24 | 97.09% | 0.090288 | 0.044485 | 0.012982 | 0.969642 | 90.66% | 83.91% |

### Audio selected per-slot metrics (calibrated)

| Slot | Options | Count | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| command | 4 | 889 | 99.44% | 0.014153 | 0.007617 | 0.006088 | 0.991333 |
| color | 4 | 889 | 99.33% | 0.016037 | 0.008982 | 0.006234 | 0.990594 |
| preposition | 4 | 889 | 96.18% | 0.093988 | 0.054972 | 0.018571 | 0.972634 |
| letter | 25 | 889 | 93.48% | 0.192133 | 0.093311 | 0.017151 | 0.949229 |
| number | 10 | 889 | 99.55% | 0.019199 | 0.008432 | 0.009713 | 0.987717 |
| adverb | 4 | 889 | 99.66% | 0.006313 | 0.003454 | 0.004415 | 0.996422 |

### Video selected per-slot metrics (calibrated)

| Slot | Options | Count | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| command | 4 | 889 | 41.39% | 1.262191 | 0.690441 | 0.052364 | 0.429364 |
| color | 4 | 889 | 32.73% | 1.356030 | 0.734974 | 0.031793 | 0.295541 |
| preposition | 4 | 889 | 37.91% | 1.310786 | 0.713356 | 0.050663 | 0.338985 |
| letter | 25 | 889 | 6.86% | 3.166561 | 0.955716 | 0.002628 | 0.066291 |
| number | 10 | 889 | 16.99% | 2.196167 | 0.879135 | 0.025571 | 0.174161 |
| adverb | 4 | 889 | 44.99% | 1.149284 | 0.631956 | 0.073800 | 0.489777 |

### Joint selected per-slot metrics (calibrated)

| Slot | Options | Count | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| command | 4 | 889 | 98.76% | 0.034952 | 0.017628 | 0.012044 | 0.979974 |
| color | 4 | 889 | 98.43% | 0.037230 | 0.022726 | 0.006726 | 0.987283 |
| preposition | 4 | 889 | 96.74% | 0.107324 | 0.056542 | 0.019751 | 0.953295 |
| letter | 25 | 889 | 90.66% | 0.297411 | 0.141347 | 0.025398 | 0.914252 |
| number | 10 | 889 | 98.54% | 0.035490 | 0.017737 | 0.008237 | 0.983485 |
| adverb | 4 | 889 | 99.44% | 0.029319 | 0.010927 | 0.005734 | 0.999562 |

### Raw best versus raw final

| Arm | Checkpoint | Epoch | Accuracy | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| audio | best | 24 | 97.94% | 0.062964 | 0.030397 | 0.015986 |
| audio | final | 24 | 97.94% | 0.062964 | 0.030397 | 0.015986 |
| video | best | 3 | 30.15% | 1.746739 | 0.770064 | 0.051902 |
| video | final | 24 | 31.23% | 2.198649 | 0.880648 | 0.267778 |
| joint | best | 24 | 97.09% | 0.092080 | 0.045001 | 0.018796 |
| joint | final | 24 | 97.09% | 0.092080 | 0.045001 | 0.018796 |

### Fixed train/validation raw learning curves

| Arm | Epoch | Train Accuracy | Val Accuracy | Train NLL | Val NLL | Val Brier | Val ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| audio | 1 | 77.98% | 77.05% | 1.029079 | 1.040979 | 0.410314 | 0.271029 |
| audio | 4 | 94.08% | 90.64% | 0.294849 | 0.360476 | 0.143192 | 0.087581 |
| audio | 8 | 99.03% | 95.89% | 0.090299 | 0.159827 | 0.068682 | 0.053311 |
| audio | 12 | 99.82% | 97.32% | 0.038704 | 0.102520 | 0.045555 | 0.034939 |
| audio | 16 | 99.98% | 97.71% | 0.020559 | 0.080726 | 0.037098 | 0.026008 |
| audio | 20 | 99.99% | 97.86% | 0.012595 | 0.069583 | 0.032851 | 0.020010 |
| audio | 24 | 100.00% | 97.94% | 0.008455 | 0.062964 | 0.030397 | 0.015986 |
| video | 1 | 30.38% | 27.97% | 1.773177 | 1.792270 | 0.786361 | 0.041877 |
| video | 3 | 37.49% | 30.15% | 1.638392 | 1.746739 | 0.770064 | 0.051902 |
| video | 4 | 39.92% | 29.92% | 1.587814 | 1.762974 | 0.777512 | 0.081256 |
| video | 8 | 45.86% | 31.10% | 1.456871 | 1.832071 | 0.800876 | 0.147067 |
| video | 12 | 49.30% | 31.05% | 1.378076 | 1.963315 | 0.829188 | 0.193505 |
| video | 16 | 51.79% | 31.14% | 1.318804 | 2.056281 | 0.851880 | 0.229458 |
| video | 20 | 53.28% | 31.48% | 1.278912 | 2.124567 | 0.863601 | 0.246266 |
| video | 24 | 54.79% | 31.23% | 1.240898 | 2.198649 | 0.880648 | 0.267778 |
| joint | 1 | 76.61% | 75.01% | 1.028895 | 1.054602 | 0.417977 | 0.243900 |
| joint | 4 | 94.56% | 89.60% | 0.283190 | 0.387361 | 0.155811 | 0.087329 |
| joint | 8 | 99.31% | 94.79% | 0.079989 | 0.188158 | 0.083124 | 0.052215 |
| joint | 12 | 99.93% | 96.08% | 0.031912 | 0.131464 | 0.060552 | 0.035272 |
| joint | 16 | 100.00% | 96.57% | 0.016442 | 0.109951 | 0.052143 | 0.025890 |
| joint | 20 | 100.00% | 96.87% | 0.009985 | 0.099069 | 0.047851 | 0.020676 |
| joint | 24 | 100.00% | 97.09% | 0.006686 | 0.092080 | 0.045001 | 0.018796 |

### Fixed selected-joint controls

| Condition | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | --- | --- | --- | --- | --- |
| matched | 97.09% | 0.090288 | 0.044485 | 0.012982 | 0.969642 |
| audio_mismatch | 31.36% | 11.704803 | 1.328633 | 0.654913 | 0.968216 |
| video_mismatch | 96.91% | 0.095985 | 0.047533 | 0.011304 | 0.967195 |
| audio_missing | 24.99% | 2.755128 | 0.974429 | 0.313002 | 0.562545 |
| video_missing | 97.11% | 0.088358 | 0.043032 | 0.010886 | 0.968487 |

### Selected calibrated plain/Lombard condition groups

| Arm | Condition | Utterances | Accuracy | NLL | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| audio | l | 450 | 98.00% | 0.056018 | 0.028476 | 0.011790 |
| audio | p | 439 | 97.87% | 0.057947 | 0.030472 | 0.013088 |
| video | l | 450 | 30.96% | 1.734783 | 0.764369 | 0.047730 |
| video | p | 439 | 29.31% | 1.745692 | 0.770905 | 0.045290 |
| joint | l | 450 | 96.96% | 0.091509 | 0.045059 | 0.016956 |
| joint | p | 439 | 97.23% | 0.089035 | 0.043896 | 0.016240 |

## Contributions and uncertainty

Selected joint weights and their calibration temperatures remained fixed across
matched, audio/video mismatch and absent-stream controls. Donor permutations are
label-independent bijections, preserve plain/Lombard condition and use another
speaker group. Audio mismatch substantially degrades predictions. Video mismatch
changes macro Accuracy by−0.187pp, interval[−0.760,+0.431], NLL by+0.005697,
interval[−0.008479,+0.019948]. This does not establish useful visual contribution.
Removing video changes Accuracy by+0.019pp, interval[−0.594,+0.780]; removing audio
severely degrades it. Missing masks were unseen during training: those controls
are diagnostic OOD, not validated missing-input support.

All differences use10,000 paired whole-speaker-group resamples, seed17, pooled
metrics recomputed, linear percentile95% intervals,9 groups/889 utterances.
Accuracy/NLL/Brier/ECE/minimum-slot/all-six-correct differences are saved. The
same development validation selected checkpoints; calibration used a separate
split. Intervals do not remove selector bias or prove blind/generalization
performance. Mismatch donors couple observations across groups, so mismatch
intervals are diagnostic and do not have full independent-group guarantees.

## Learning and bottlenecks

Video raw validation NLL is best at3(1.746739), worsens to2.198649 at24 while
fixed-train NLL falls to1.240898. Final train Accuracy54.79% versus validation
31.23% shows clear overfitting. Its final checkpoint is retained, not substituted
for the selected3. Calibration of final with best's temperatures still has worse
NLL than selected. Do not extend this pooled video run to rescue its metric.

Audio and joint validation NLL keep improving through24; their final raw
fixed-train versus validation Accuracy gaps are2.058pp and2.906pp. They are not
proven clean plateaus or free of memorization. Joint fits train100% yet generalizes
worse than audio; adding visual features increased available nuisance information
without measured benefit. No automatic epoch extension is justified by that.

The underlying audiovisual input is real and synchronized, but spoken lexical
labels are already strongly determined by audio. This task does not establish
that its visual branch contains necessary complementary evidence. Global pooling
removes chronological token access from the readout; the pretrained temporal
encoder can still encode order, so this is **not** proof of raw-video order
invariance or of a universal VideoPrism capacity ceiling. Generic visual
representation versus fine lip-reading, spatial/temporal pooling, visible speech
ambiguity and readout capacity are unresolved explanations, not established causes.

Retain the audio model as a narrow reusable research reference. The next useful
work is to distinguish those explanations and define genuinely complementary,
licensed native visual/AV decisions before additional fusion training. Inspect
retained chronological tokens before proposing a separately frozen readout
experiment; do not enlarge or retrain encoders, relax parity, change this corpus,
or quantize an unqualified product. General conditional text/image/video quality
and the95–98% four-modality objective remain open.

## Execution, resource accounting and artifacts

Training/report source commit:
`725c0660879c6c33f0612a5f30e22c45212a0cee`. Actual commands from the isolated
local controller, cwd`C:/CodexWorktrees/pretrained-reuse-omni`:

```powershell
python -m scripts.train_lombard_grid_readouts
python -m scripts.evaluate_lombard_grid_readouts
```

Both exit0; final controller`PASS_NATIVE_THREE_ARM_STUDY`, trainer
`PASS_THREE_ARM_EXECUTION`, evaluator`PASS_FIXED_MODEL_REPORT`. The controller
waited for the exact original process's zero exit, completed hashes/loader gates,
clean unchanged commit, idle GPU and disk reserve. It did not restart the encoder.
Its exact wrapper commands/timestamps and SHA are in the result JSON. Normalized
JSON contains all options; option permutation and same-backend reload are exact.
These are fresh head runs; no strict optimizer-resume claim is made.

Native video extraction:2026-10-13 17:15:09–20:36:02UTC, **12,052.914s(3.348h)**,
46 full-token replay checks and4 native reference matches, peak working set
4,335,759,360bytes. Native source commit at launch`bbd194e`, source config and
runner SHA unchanged; exact base source`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`,
checkpoint SHA`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
Before/after parameter SHA match; all5,367 media identities/features reload.

Prior full audio extraction160.331s. Cached-head three-arm training window
13.123s(each arm4.691/3.228/3.551s); fixed-model evaluation3.970s. Child-process
wall time, including imports/loader gates, is17.115s/7.775s.
Per-arm windows include validation, calibration and saves; they are not measured
GPU-active optimizer hours. Local extraction/training/report wall-time sums do
not include prior source acquisition, diagnostics or mechanical fixtures.
Environment: local RTX3080 Laptop,Python3.11.9,Torch2.6.0+cu124,CUDA12.4,
NumPy2.4.6,FP32,TF32 off, deterministic
eager matmul. Native extraction uses JAX0.6.2/TF2.18.1/NumPy1.26.4/CPU with the
existing approved system Python. No dependency upgrades or cloud compute; cost0.
Cached-head peak Torch allocation is127,351,808bytes(reserved132,120,576), excludes
encoders/decoder/media extraction and is not deployment-memory evidence.

Resident component counts including head and2,304 normalization coefficients:
audio8,287,923; video114,444,723; joint122,653,107. Each selected head file310,840
bytes; separate final310,848bytes, normalizer10,250bytes. These exclude large
retained encoders; no combined deployment package,<=200MB package, end-to-end
latency or95–98% Omni product has been qualified. No ternary/QAT/compression.

Local artifacts beneath`C:/CodexArtifacts/pretrained-reuse-r2/`:

- `lombard-grid-videoprism-native-features-01/`: complete native metadata,
  runner/effective config, framewise/global features and hashes.
- `lombard-grid-whisper-features-01/`: complete speech feature bank.
- `lombard-grid-pooled-readouts-01/{audio,video,joint}/`: both checkpoints,
  24-epoch history,768 optimizer updates, sampler, config/provenance, calibration,
  all per-example raw/calibrated best/final logits/probabilities/JSON.
- `lombard-grid-pooled-readouts-report-01/`: all replays, slot/condition/speaker/group
  metrics, six uncertainty comparisons and frozen mismatch/missing controls.
- `lombard-native-comparison-controller-01/`: exact commands, timestamps,
  source hashes, exit status and untouched child logs.

All legacy Teachers, datasets, artifacts and original failed/smoke/mechanical
fixtures are retained. Only small reports/curves enter Git; no media or weights.
Local verification after recording results: **953 passed,1 skipped** in14.86s,
Ruff passed,25 manifest validation/audit commands passed. CSV and result JSON
line endings are preserved explicitly so their recorded byte hashes survive
checkout on other platforms.

## Review and scope rulings

One independent whole-code review was completed before native learning while
the CPU extraction was live. No Critical/Minor issues; the one Important saved
NumPy/checkpoint replay gap was fixed with24 RED→GREEN cases, full953-test suite
and mechanical same-backend fixture replay. No second review pass. Final native
interpretation was outside that review; this author checked the complete native
report, every learning point, controls and declared accounting after terminal exit.

Execution rulings: implement pure Task4 helpers while Task3 waited(cost: initial
integration unverified); preserve p/l for different-group donors(cost: inadequate
diversity fails rather than relaxing); separate pure metrics from IO evaluator
(cost: one extra module); review before native quality(cost: no independent review
of final interpretation); finite protected handoff controller(cost: changed source
or occupied GPU fails closed, no automatic restart). No deferred Minor findings.
These rulings do not change data, optimizer, budget, vocabularies or success target.
