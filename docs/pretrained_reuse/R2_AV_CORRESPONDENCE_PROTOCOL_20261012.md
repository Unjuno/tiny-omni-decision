# Prospective audio-to-video correspondence probe

## Scope and motivation

The visibility proposal failed its query-only control (96.44% without media).
Do not fit it. The next bounded candidate asks which of four silent video
observations originates from the query audio's recording. No sound-class label,
ground-truth object identity, visibility, filename or recording ID enters the
scorer. Correctness is recording correspondence, not a claim that all sounds
are visually identifiable or that recordings cannot sound alike.

This is a task-matched AV alignment diagnostic within R2-M. It does not replace
the required general text/image/audio/video decision battery, nor demonstrate
that sensors or optional natural-language conditions are solved.

## Freeze before acquisition and fitting

- Source: official Perception Test archive generations, pinned by manifest.
  Retain CC-BY attribution and exact member hashes. No test/audit archive.
- Development acquisition ceiling: 256 official-train and 200 official-valid
  recordings, selected by SHA-256(`17:video_id`) ascending, independent of labels.
  First smoke: eight train recordings only. Previously fetched matching files
  may be reused after hash verification. Never substitute shorter/easier clips.
- Keep official split assignments; upstream groups script × participant, not
  participant alone. Do not random-split official train into validation.
- Within each split, partition the fixed ordered IDs into quartets. Query all
  four recordings' audio against the same four videos. Each video is correct
  exactly once per quartet. Assign each of the four correct positions once per
  quartet using seed17-derived hashes; order distractors separately per query.
  Score content only. ID strings remain routing metadata.
- Failed decoding, missing sound and semantically ambiguous correspondence stay
  explicitly reported. No quality-based exclusion or audio class balancing.
- Video: existing eight-frame frozen VideoPrism profile; audio: frozen DyMN.
  Use the common observed audio/video interval and actual timestamps; no labels
  or trailing answer-reveal frames. Resolve cut handling before any scored run.
  No hidden per-clip resampling or encoder changes based on outcomes.
- Query audio and candidate videos are separate observations. Neither metadata
  identity nor candidate position may be provided to the learned score function.

## Required controls and interpretation

Untrained fixed projections, train-fitted projection, audio-zero, video-zero,
shuffled-query audio, reversed candidate order and one-to-one ID/hash checks.
Chance for the balanced four-candidate construction is 25%; observed control
scores and calibration must still be measured. Balance does not rule out
recording/environment shortcuts. Same sound classes and repeated scripts may
make mismatched recordings acoustically plausible; preserve these cases.

First fitting candidate, only after the acquisition/rights/timing gates pass:
seed17, frozen encoders, 64-dimensional bias-free projections for 384D audio and
768D video, normalized dot-product logits with fixed temperature0.1. AdamW
lr0.001, weight decay0.01, batch of quartets, 100 epochs. Select earliest minimum
validation NLL, retain final separately. No extra seeds, schedules or capacity
changes in this initial comparison. A smoke may check mechanics on train only;
training accuracy from that smoke is not quality evidence.

Report Accuracy/NLL/Brier/ECE, source recording counts, quartet-grouped bootstrap
10,000 draws seed17, all attempted configurations, best/final reload, latency,
memory and total resident parameters. Quartets couple candidate errors and are
the minimum resampling unit; participant overlap remains an additional limitation.
This protocol does not authorize claiming 95–98% general quality from retrieval.

## Current status

Protocol fixed before new media acquisition or readout fitting. Source admission
and media/cut handling are still required. Existing two-video feature artifacts
are preserved; their event-level visibility windows are not silently repurposed
as recording-level retrieval examples. Do not train or score this task until the
correspondence records pass the stated gates.

## Executed manifest construction

The deterministic builder produced 256 train routing records in 64 quartets from
the pinned train annotation keys. Reversing input enumeration leaves the result
unchanged. Tests verify exact positive identity, identical candidate sets within
each quartet, balanced correct positions and rejection of test split requests.
The JSON SHA-256 is
`560f3bc7235c842746038e6e5a886471f4dc08cc515d9eba4d12968e608f393b`,
stored at `perception-correspondence-plan-01/train-routing.json` under the R2 root.
This file contains supervision/provenance and must not be embedded as model text.

The first smoke IDs are video_3174, video_7855, video_9027, video_1688,
video_4873, video_1693, video_2797 and video_5013. No validation IDs or labels were
inspected to choose them. Only eight-clip acquisition is initiated in this stage;
the remaining planned train records do not imply downloaded media or training.

## Eight-clip acquisition completed

`python -m scripts.probe_perception_media --count 8 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-media-probe-02`
completed download/reuse, CRC/SHA checks and full audio/video decoding of the eight
fixed IDs. Existing first two files were copied only after hash/CRC validation;
their originals remain untouched. Archive generation and size checks are the
same as the preceding pilot. All decoded streams have increasing timestamps.

Metadata is not a substitute for decoder evidence: video_5013 decodes 699 frames
while annotations list 700. Sound sample counts also differ for some clips.
The next preprocessing stage must use decoded PTS, the observed common interval
and upstream cuts; do not fabricate a missing frame or pad audio to metadata.
Detailed counts, file hashes and timing are in manifest.json and timing.json.
Eight train recordings are not independent evidence for general quality.

No fitting or validation acquisition occurred. Protocol/builder/acquisition code
was committed as a42ebd3; 604 CPU tests passed and both CI runs succeeded.

## Common-time inputs and actual-rate correction

All eight fixed recordings now have input-only audio/video windows. Bounds use
the intersection of decoded audio support and video PTS, ending conservatively
before the last video PTS and any upstream cut. Sample supports round inward;
video samples are selected only inside the resulting half-open interval. Payloads
contain PCM and RGB arrays, with routing/targets retained separately.

Actual decoding found video_4873 is **44.1 kHz**, despite annotation metadata
saying48 kHz. Its earlier sample-count difference cannot be interpreted as lost
audio without rate conversion. The other seven streams are48 kHz. The preparer
uses each decoded rate and exact rational resampling ratio to24 kHz; it rejects
changing rates, missing clocks, nonplanar/nonfloat PCM and unavailable intervals.
The audio encoder then retains its existing24-to32 kHz preprocessing profile.

The fixed DyMN and VideoPrism encoders each extracted eight finite feature
vectors; duplicate execution gave maximum absolute difference zero for both.
This is feature plumbing, not learned alignment or validation quality.
Raw media and all previous feature runs remain unchanged.

Artifacts under R2: `perception-correspondence-windows-01`,
`perception-correspondence-audio-01`, `perception-correspondence-video-01`.
Reports record actual rates, interval bounds, selected rational PTS, cuts, window
hashes and pinned model identities. The new preparation command is
`scripts.prepare_av_correspondence_windows --output <fresh>` in the existing
PyAV environment with the worktree src on PYTHONPATH. Feature extraction uses
`scripts.extract_perception_pilot --windows <new-result.json> --modality audio|video --output <fresh>`.
No readout fitting or validation/test access occurred. Actual44.1 kHz interval
handling has a regression test. Duration/recording-background shortcuts remain
possible and must not be presented as general semantic understanding.

## Validation identity audit and bounded smoke admission

Official validation metadata was pinned to storage generation1688372812971719
and SHA256 `eeefeb43955b1850b5e521ad774fe66675baac0a2a06617bd83ffbe43aaf03d7`.
It contains5,900 official-valid recordings and zero recording-ID overlap with
all2,184 train keys. The predeclared hash ordering selects200 recordings /50
quartets, routing SHA256
`03d389b0eda847b4d16e0205a7c285ad78ebecfb6405fde1c85c0ddfd569d395`.
The selected media total2,525,646,586 bytes; maximum member122,791,611 bytes.
Eleven selected recordings need upstream cuts. None were removed or substituted.
The previous32MB pilot member cap cannot acquire this fixed selection; a new
bounded acquisition must budget for these actual sizes. Metadata inspection has
not acquired validation media or measured validation performance.

Evidence lives in `perception-correspondence-valid-audit-01`: pinned URLs/ETags,
annotation/cut hashes, all selected members, routing and result. Validation media
is generation1685479088600941; complete archive75,375,143,328 bytes. Only ZIP index
ranges were read. Content-level cross-split checks remain pending acquisition.
Official script-by-participant splitting does not guarantee unseen participants.

The existing eight train recordings are admitted only to the protocol's technical
projection smoke: official CC-BY-4.0 source and attribution retained, media hashes
verified, actual clocks/cuts resolved, no supervision in feature payloads. This
narrow admission does not reverse the rejected visibility-task decision or admit
a quality benchmark. Use seed17, two quartet batches per epoch,100 epochs of the
fixed73,728-parameter projection. No validation selector is used in this train-only
smoke. Keep final weights, history, all control predictions and reload evidence;
training memorization is not generalization. Encoders are not loaded or updated.
Full development fitting still requires independent validation media/content gates.

## Executed projection mechanics smoke

Source commit41f1e3bdc2c9dc81e1a5e508785b166a4a2c8322 ran
`python -m scripts.smoke_av_correspondence --output C:/CodexArtifacts/pretrained-reuse-r2/perception-correspondence-smoke-01`.
CPU, seed17,100 epochs /200 optimizer updates, eight unique recordings /two
quartets /800 presentations. Only73,728 projection parameters were trained;
no encoder loaded. Fit/history measurement took0.360 seconds, excluding file
loading and earlier feature extraction. Final checkpoint296,332 bytes, SHA256
`526db7760da7d5ed7774f2e5634cf024b3548b93fb3382a5e61b31652a6fe809`.

| Train-only condition | Accuracy | NLL | Brier | ECE15 |
|---|---:|---:|---:|---:|
| Untrained fixed projection |0.625|1.343219|0.728032|0.402192|
| Trained final |1.000|0.000061|0.000000006|0.000061|
| Audio zero |0.250|1.386294|0.750000|0.000000|
| Video zero |0.250|1.386294|0.750000|0.000000|
| Query audio cyclically shifted within quartet |0.000|11.234765|1.999845|0.999939|
| Reversed options with remapped targets |1.000|0.000061|0.000000006|0.000061|

Reload logits were bit-exact; reversal logits pass1e-6 tolerance. All feature and
routing hashes remained unchanged. The zero controls are uniform by construction
of bias-free projections, so their scores verify mechanics rather than proving
semantic complementarity. All probabilities/targets, config/hashes,100-point
history and executed source are retained. There is no selected best checkpoint,
validation, confidence interval or quality claim: fitting eight examples can
memorize recordings. The initial62.5% is only5/8 examples and is not reliable
zero-shot evidence. The next stage is the fixed256-train/200-valid acquisition
and independent development comparison, not further fitting on these eight.

Ruff and606 CPU tests passed (one optional test skipped). Source attribution and
all original input/model artifacts are preserved; no GPU or paid compute used.

## Fixed development media acquisition started

The combined plan SHA256 is
`0c6a07048c393d2fddd54fb3e72883055bb43634f9e3a8cac5d8570edcc39931`.
All256 train and200 valid selections total5,938,164,261 bytes. The bounded loader
raises on missing/duplicate members, wrong split, changed ZIP sizes/CRC, or disk
reserve failure. It retains an8GB free-space reserve and a256MB per-member cap;
no recording is substituted to satisfy that cap. Storage generations, exact
members, cuts and metadata are in `perception-correspondence-acquisition-plan-01`.

Source1091148 launched separate train and valid processes into fresh directories:
`perception-correspondence-train-media-01` and
`perception-correspondence-valid-media-01`. The first eight train files reference
verified existing pilot files instead of copying them. Each downloaded member is
ZIP-CRC checked, size checked and SHA256 hashed; an incomplete file retains its
`.partial` suffix. Each completed record is logged immediately in manifest.json.
This entry records launch, not completion, decoding, content-overlap clearance or
training. No test/audit archive is accessed. The source and selected metadata are
fixed; this is not annotation- or performance-based acquisition.

Command template:
`python -m scripts.acquire_av_correspondence --plan <plan.json> --plan-sha256 0c6a07048c393d2fddd54fb3e72883055bb43634f9e3a8cac5d8570edcc39931 --split train|valid --output <fresh>`.
Local checks: Ruff passed;607 CPU tests passed and one optional test skipped.

## Development preprocessing parity gate

The pilot decoder is now a reusable `decode_window` with unchanged clock,
resampling and frame-selection operations. In a separate PyAV process, all eight
pilot PCM arrays, RGB arrays and timing fields match their original artifacts
exactly. Evidence: `perception-correspondence-preprocessor-parity-01/result.json`.
The saved original windows and features were not overwritten.

`scripts.prepare_correspondence_development` requires both complete256/200 media
manifests and the fixed plan hash before creating output. It verifies recording
IDs, split labels and every media hash, rejects cross-split exact media duplicates,
and then prepares the same common-clock windows. Exact prepared-array content is
also checked across splits. These exact hashes do not establish absence of near
duplicates, recurring scripts, participants or environments.

A live negative check against the unfinished acquisition correctly rejected it
with `train acquisition incomplete` and did not create an output directory.
Full preparation has therefore not started. When acquisition finishes, run the
new script in the existing PyAV environment with a fresh output directory. No
subsampling, failure-based replacement or validation-label filtering is allowed.
Ruff and607 CPU tests pass; the PyAV parity and incomplete-input checks were run
separately from normal CI. Download processes continue unchanged.

## Acquisition completed; full preprocessing running

Both acquisition processes exited0. Final manifests report
`PASS_ACQUISITION_NOT_DECODE`: train256 files /3,412,517,675 bytes, valid200 files /
2,525,646,586 bytes. Wall times were130.881 and101.398 minutes respectively,
with overlapping execution; do not sum them as elapsed wall time. No selections
were replaced. These counts include the eight verified reused train files.

Full preprocessing was launched from ef06dcc into
`perception-correspondence-development-windows-01` using the existing PyAV Python
and `python -m scripts.prepare_correspondence_development --output <fresh>`.
It passed the initial complete-manifest, recording identity, all-media SHA256 and
exact cross-split media-duplicate gates and began producing windows. This records
launch and initial gates only: full decode, prepared-content checks, feature
extraction and development comparison remain incomplete until their reports pass.
