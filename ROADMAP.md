# Roadmap — Pretrained-Reuse Omni Decision (proposed 2026-10-10)
**2026-10-31 05:23 JST R2-M metadata-only file inventory:** The public Mendeley snapshot confirms version 3/CC BY 4.0; the root file endpoint lists only seven `.m` files and one `.txt` file (20,631 bytes total). Nested files remain unknown because folders require OAuth; no media payload or zip endpoint was accessed. This narrows, but does not resolve, source availability; keep the candidate unadmitted. The metadata endpoint took 2.484/1.123/1.113 seconds across three requests (mean 1.573 seconds), which is not a media-download estimate. See the [file inventory follow-up](docs/pretrained_reuse/R2_HUMAN_HANDOVER_VIDEO_IMU_SOURCE_SCREEN_20261031.md#metadata-only-file-inventory-follow-up).
**2026-10-31 05:18 JST R2-M human-handover source screen:** Mendeley Data v3 is explicitly CC BY 4.0 and describes 1,087 synchronized video/sensor recordings from 18 volunteers, but its subjective four-point ratings have documented interpretation differences and a both-input-required supplied-option task plus participant-disjoint split are not established. Keep `REVIEW_NOT_ADMITTED`; no payload or labels were accessed. See [the source/task screen](docs/pretrained_reuse/R2_HUMAN_HANDOVER_VIDEO_IMU_SOURCE_SCREEN_20261031.md).
**2026-10-31 05:10 JST R0 live refresh:** Remote `main` is `09246d7b`; the roadmap branch is clean at `e3832c74` and matches its remote. PR #15 remains open/draft and CI run #41120472190 is green. The local RTX 3080 is idle, C: has 30.37 GB free, and the primary user worktree still has pre-existing uncommitted changes that were left untouched. R2 still lacks an admitted independent task requiring multiple streams; do not train on the exposed Perception transfer or underpowered image expansion. See [the read-only live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261031_0510JST.md).
**2026-10-31 R1 MOMENT runtime recheck:** The pinned MOMENT-1-small FP32 checkpoint (37,901,512 tensor parameters; the initialized embedding model contains 35,337,408 parameters; checkpoint SHA-256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`) loaded from the audited local `momentfm` source and produced finite `[1,6,16,512]` embeddings on the RTX 3080 Laptop. The first forward took 0.4865 s and a separate immediate replay took 0.2189 s; peak allocated VRAM was 162.7 MB. This was an all-zero runtime smoke only; no task data or quality metric was used. No package was installed; a process-local no-op `dotenv` shim enabled the offline local load. MOMENT sensor task quality remains open. Do not reuse the already-exposed three-subject HARTH validation for another fit; wait for an admitted fresh group split. See [the runtime recheck](docs/pretrained_reuse/R1_MOMENT_RUNTIME_RECHECK_20261031.md).
**2026-10-31 R2-M MUN-FRL source screen:** The official raw-data grant is CC BY 4.0 and provides synchronized RGB/IMU with RTK ground truth, but only nine independent flight sequences and approximately 435.5 GB across the full release. This is a possible aerial visual-inertial task, not an admitted quality benchmark; no payload was accessed. Keep it REVIEW and do not train on the small workshop subset. See [source/task screen](docs/pretrained_reuse/R2_MUN_FRL_VISUAL_INERTIAL_SCREEN_20261031.md).
**2026-10-31 R2-M implementation increment:** Added a text-free, candidate-equivariant frozen-feature Decision readout, a fixed-schema numeric/enum option encoder, and a single-modality adapter that reads a SHA-256-checked aligned cache, enforces query-time freshness, and abstains on missing/stale evidence without importing a language model. Its CPU tests pass; task-specific leakage audit and measured M1 evaluation remain open. See [implementation note](docs/pretrained_reuse/R2_TEXTFREE_NATIVE_FEATURE_READOUT_20261031.md).
**2026-10-31 R2-M native readout diagnostic:** A fixed 10-epoch typed-option image readout fit 49,793 parameters on 1,145 frozen-feature train rows and reached 75.35% Accuracy / 0.7093 NLL on 430 previously exposed development rows. Paired observer-group intervals show Accuracy remains uncertain while NLL/Brier/ECE improve; M1 and multimodal gates remain open. No encoder was loaded or updated during the fit; no sealed audit or GPU training was used. See [the diagnostic](docs/pretrained_reuse/R2_NATIVE_FEATURE_DECISION_DIAGNOSTIC_20261031.md).
**2026-10-31 R2-M FAIR-Play screen:** Three train-only pairs verified the pinned CPU VideoPrism and guarded local EfficientAT feature paths, but the released validation fold has only 187 IDs and no independently verifiable scene-group mapping. Cross-fold asset overlap prevents treating folds as independent replicates. No bulk acquisition or fitting; source remains unadmitted. See [the source screen](docs/pretrained_reuse/R2_FAIR_PLAY_SOURCE_SCREEN_20261031.md).
**2026-10-31 R2-M recent-source terms screen:** ActionSense and OctoNet publish CC BY-NC-SA/NC data terms, while CUHK-X raw data require a non-commercial DUA; none is admitted for this product-directed training path. No payload was accessed. See [the terms/task screen](docs/pretrained_reuse/R2_RECENT_MULTIMODAL_SOURCE_TERMS_SCREEN_20261031.md).
**2026-10-31 R2-M Action Verb Corpus screen:** The OFAI page labels AVC CC BY 4.0, but its Zenodo record labels the same audio/video/motion data CC BY-NC-SA 4.0; the corpus has 46 episodes / 12 people and does not establish a both-required task or grouped split. Keep REVIEW-only; no payload was accessed. See [the source/task disposition](docs/pretrained_reuse/R2_AVC_SOURCE_LICENSE_DISPOSITION_20261031.md).
**2026-10-31 R2-M iNaturalist ascending-window extension:** A metadata-only query of 8,000 records across eight taxa froze 1,813 source groups, but after removing all previously used observer/observation/photo identities, only 15 train and 5 validation Mallard records remained. It fails the frozen 700/200 and all-eight-taxa expansion gate; no media, features, evaluation or fitting followed. The eight-taxon cohort cannot be extended through this API window; see [the coverage gate](docs/pretrained_reuse/R2_INATURALIST_ASCENDING_COVERAGE_20261031.md).

**2026-10-31 02:51 JST R0/R1/R2 cross-branch recheck:** PR #15 remains Draft
at `66810de2`; the RTX 3080 Laptop is idle. The exact V-JEPA 2.1 checkpoint
still lacks an explicit checkpoint-specific license grant, while VideoPrism-B
remains the admitted Path A video source. The fresh-observer iNaturalist
readout reached 74.88% Accuracy on its narrow eight-species image task but
showed overfitting; pages 41–50 produced only 27 train / 14 validation rows and
fail the frozen expansion gate. Perception joint AV remains below video-only
Accuracy, so R2 M-2 is still open. No new training is justified by these
gates. See [the cross-branch recheck](docs/pretrained_reuse/R0_R1_R2_CROSS_BRANCH_RECHECK_20261031.md),
[V-JEPA rights recheck](docs/pretrained_reuse/R1_VJEPA21_LICENSE_RECHECK_20261030.md),
[iNaturalist readout](docs/pretrained_reuse/R2_INATURALIST_FRESH_VALIDATION_READOUT_20261029.md),
and [iNaturalist coverage gate](docs/pretrained_reuse/R2_INATURALIST_PAGES_41_50_COVERAGE_20261030.md).

**2026-10-31 R2-M EgoViz cross-branch result:** PR #44 reports one fixed
checkpoint evaluation on 2,394 timestamp-correspondence choices from 69 fresh
session groups. It is useful development evidence for video+IMU temporal
alignment, but the GRU's NLL/ECE are poor, participant identity across sessions
is unknown, and the target is not a natural action label. Keep M-2 open; do not
re-evaluate or tune on those groups. See
[`R2_EGOVIZ_PR44_CROSS_BRANCH_RECONCILIATION_20261031.md`](docs/pretrained_reuse/R2_EGOVIZ_PR44_CROSS_BRANCH_RECONCILIATION_20261031.md).

**2026-10-31 02:11 JST R0 Orbitune run terminal check:** Both tracked
processes (worker `3668`, wrapper `33576`) are absent. The RTX 3080 reported
0% utilization / 0 MiB used. The final saved checkpoint is step `586076` /
`14,268,978,580` events, 17,641 above the command's `14,268,960,939` stop
threshold; this is consistent with threshold completion. The current-run exit
code/reason was not persisted, so the exact termination cause remains UNKNOWN.
Best stays step `521917`; final saved validation is `-1.576682` at step
`585754`; no non-finite loss/gradient counters. See [02:11 terminal audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261031_0211JST.md).

**2026-10-31 01:44 JST R0 checkpoint update:** PID `3668` remains the live
GPU process. The saved checkpoint is now step `585395` / `14,250,013,261`
events; latest validation is `-1.601044` at step `585036`, while best remains
step `521917` / `-1.627876`. Five observed save intervals average about
13.77k events/s. The run has `18,947,678` saved-count events to its configured
stop threshold, an estimated 23 minutes at that mean; threshold is expected
before the next 30M-event checkpoint. Current live progress remains unknown.
See [01:44 audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261031_0144JST.md).

**2026-10-31 01:03 JST R0 checkpoint update:** Orbitune PID `3668` is still
GPU-visible. The saved checkpoint advanced to step `584321` / `14,220,006,898`
events at 01:03:18; latest validation loss is `-1.510132` at step `584321`,
and best remains `-1.627876` at step `521917`. Four saved intervals now
average about 14.15k events/s. About 48.95M saved-count events remain; this
projects about 58 minutes to the configured threshold and about 35 minutes to
the next 30M-event save, assuming the measured rate persists. The current
unsaved count remains unknown. See [01:03 audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261031_0103JST.md).

**2026-10-31 00:27 JST R0 checkpoint recheck:** The same GPU-visible PID
`3668` remains active. `model.pt` and `model.healthy.pt` advanced to step
`583245` / `14,190,001,306` events; latest saved validation is step `582886`
with loss `-1.551756`; best remains step `521917` / `-1.627876`. Three saved
intervals measured about 13.31k, 14.90k and 14.89k events/s (mean 14.37k),
supporting a roughly 35-minute next 30M-event checkpoint estimate. About
78.96M saved-count events remain to the configured limit, roughly 1h32m at
that observed mean. Estimates are not live counters. No process or data was
changed. See [00:27 checkpoint audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261031_0027JST.md).

**2026-10-31 00:03 JST R0 Orbitune checkpoint update:** The same live CUDA
process PID `3668` remains active. A new checkpoint was saved at 23:52:37 JST:
step `582168`, `14,160,017,296` events; best remains step `521917`, and latest
validation is `-1.511443` at step `582168` (four batches / 98,304 events).
No non-finite loss or gradients are recorded. The command's event threshold is
`14,268,960,939`; `108,943,643` saved-count events remain. Two observed
checkpoint intervals averaged about 14.1k events/s, implying roughly 35
minutes to the next 30M-event save and 2h09m to the configured stop threshold
if that rate persists. These are projections; the trainer logs remain stale
at 21:58 and the unsaved live step is unknown. Read-only evidence and hashes:
[00:03 recheck](docs/pretrained_reuse/R0_LIVE_AUDIT_20261031_0003JST.md).

**2026-10-30 23:55 JST R0 PR synchronization:** Fast-forwarded the verified
R0 audit commit `945fd5ec572c73063a41f06e7493beccfd5089df` onto the PR #15
source branch without force. GitHub branch metadata, PR API, and
`refs/pull/15/head` now agree on that SHA. All three Actions test checks for
the synchronized PR head completed successfully. This resolves the prior
head-reference mismatch; see the [23:51 audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261030_2351JST.md)
for the incident and read-only evidence.

**2026-10-30 23:51 JST R0 process/checkpoint recheck:** PID `3668` is still
alive and listed by `nvidia-smi` as the CUDA process. Five-minute samples from
23:27–23:34 and a 16-minute watch from 23:35–23:51 showed CPU time increasing
(about 0.65 CPU-seconds per wall-second in the latter interval). The RTX 3080
sample at 23:51 was 99% GPU utilization and 15,821/16,384 MiB allocated. The
checkpoint stayed unchanged at its 23:19:03 save (step `581092`,
`14,130,003,782` events, SHA-256 unchanged); the current live step/event count
is still UNKNOWN. The files `train.log` and `watchdog.jsonl` end at 21:58; the
watchdog records a different trainer child (`19004`) exiting at step `578783`
after an idle-yield request. Do not treat that old child-exit record as proof
that the currently GPU-visible PID `3668` has stopped. PR #15's branch ref is
at `e1ca037`, and Actions has two successful runs for that SHA, but GitHub's
`refs/pull/15/head` and PR metadata still report `db19d172`; the PR-head
divergence remains unresolved. Full read-only evidence:
[23:51 recheck](docs/pretrained_reuse/R0_LIVE_AUDIT_20261030_2351JST.md).

**2026-10-30 23:22 JST R0 checkpoint-state recheck:** Orbitune's separate
Aria-MIDI/GigaMIDI job remained live (worker PID `3668`, wrapper PID `33576`).
Three five-second observations found the worker alive with CPU time increasing
by an average 3.71 seconds per 5.09 seconds wall time; `model.pt` stayed at
step `581092` / `14,130,003,782` events, last written at 23:19:03 JST. The
saved best is step `521917` (validation loss `-1.627876`); the latest saved
validation is step `580734` (`-1.121710`). The log files are stale at 21:58,
so the unsaved live step, current throughput, next checkpoint ETA and run
completion time are UNKNOWN. No process or artifact was changed. CPU checkpoint
inspection confirmed separate hashes for current, healthy, and best snapshots
and saved optimizer/sampler/RNG state; exact resume is not claimed. See the
[23:22 read-only audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261030_2322JST.md).

**2026-10-29 13:12 JST R0 checkpoint-state recheck:** Orbitune PID `22136`
is still running its separate Aria-MIDI/GigaMIDI `--steps 0` job. A CPU-only
`weights_only=True, mmap=True` read of the checkpoint saved at 13:11:51
verified step `520839` / `12,450,013,846` events; the log had reached
step `520850` / `12,450,311,232` events at 13:12:11. The checkpoint is
117,947,820 bytes with SHA-256
`E0FCD74C780872A60987706D9976F6A28F697A35F2190AA947B2224CA8961C33`.
It contains optimizer, sampler, and Python/PyTorch/CUDA RNG state; this is
checkpoint-content evidence, not a claim that resume is exact under every
runtime condition. Its 1,039-row validation history ends at step `520477`
(`-1.601522`, four batches / 98,304 events); historical best remains step
`455514` (`-1.623135`). Validation rows have no wall-clock timestamp, so the
last validation time is UNKNOWN. At about 14.6k events/s, the next
30M-event checkpoint is estimated in about 34 minutes. GPU utilization was
100% at the live sample, 15,821/16,384 MiB used, and C: had 0.374 GiB free;
no new model experiment was started because there is no safe disk reserve.
No process, data, or checkpoint was modified. See the
[checkpoint-state audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_1312JST.md).

**2026-10-29 R2 image transfer artifact recheck:** Independently replayed
checks over the preserved PR #24 prediction artifact reproduce the metrics on
all 19,988 official CLEVR-4 100k validation images (79,952 decisions), with
matching prediction/manifest hashes and normalized ten-option probabilities.
The fixed readout scores macro Accuracy 87.41%; color/shape are near 99.5%,
texture 90.28%, and count 60.44%. This is a same-family synthetic image
transfer check, not general image reasoning or joint fusion. The validation
is exposed and must not be reused for tuning. PR #24 stays separate and open;
no source branch, data, or model was modified. See the
[artifact reconciliation](docs/pretrained_reuse/R2_CLEVR4_100K_ARTIFACT_RECONCILIATION_20261029.md).

**2026-10-29 12:41 JST R0 live checkpoint and cadence recheck:** Orbitune
PID `22136` remains live with `--steps 0`; this is its separate Aria-MIDI /
GigaMIDI training job, not a Tiny Omni experiment. The live log reached step
`519850` / `12,422,446,582` events at 12:40:18, reporting `13,427` events/s.
Its 12:37:23 `model.pt` snapshot is step `519764` / `12,420,006,378` events,
117,947,372 bytes, SHA-256
`69cc9d73717cbe459cdbaf329baf58d81892a6ad321a72af21902fac258561fb`.
A CPU-only `torch.load(weights_only=True, mmap=True)` inspection confirms 1,038
validation records, optimizer and RNG state, latest validation loss `-1.614579`
at step 519,764, and historical best `-1.623135` at step 455,514. Three
15-second progress samples initially showed stale log rows; the two distinct
log updates imply about 14.4k events/s. After a roughly 20-minute controlled
wait, the next checkpoint appeared at 12:37:23, about 32 seconds earlier than
the 19m48s estimate. Current C: free space is about 0.38 GiB and RTX usage is
15,821/16,384 MiB; no process, checkpoint or dataset was changed and no new
GPU work was started. The next 30M-event checkpoint is estimated in roughly
31–34 minutes; the unbounded `--steps 0` run has no completion ETA. See the
[12:41 live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_1241JST.md).

**2026-10-29 11:29 JST R0 live checkpoint:** Orbitune PID 22136 remained live
at step 517,609 / 12,360,001,721 events. A CPU-only, mmap checkpoint read
confirmed the saved checkpoint step and full training/optimizer/RNG state; its
latest validation loss is -1.601152 at step 517,609, while the historical best
remains step 455,514 at -1.623135. `model.pt` SHA-256 is
`c7724b207f1116072d647077271022c6ad870c181df5b4a305df31016f0d5798`. No
process or run was changed. See the [live checkpoint audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_1129JST.md).

**2026-10-29 R2-M MUSIC-AVQA source screen:** The official task includes audio-
visual association and temporal questions, but its pinned train/validation
annotations share 3,808 of 3,815 validation video IDs. The official page states
CC BY-NC 4.0 and describes YouTube-origin clips; clip-level rights, a clean
group split, supplied-option construction, and both-input necessity remain
unresolved. Keep `REVIEW_NOT_ADMITTED`; no video media, test annotations, or
model outputs were accessed. See the [source screen](docs/pretrained_reuse/R2_MUSIC_AVQA_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2 Path A artifact-level verification:** The 2026-10-28 entry
already records the completed 2,048-update baseline and residual-LSTM runs.
This read-only follow-up reverified their local receipts, shared validation
IDs/source pins, exact replays and paired-audit CPU tests (3 passed), and adds
selected per-modality calibration metrics. Baseline remains selected by the
fixed hierarchical macro-NLL rule (1.6865 vs LSTM 1.6967). LSTM's selected
Video Accuracy difference is +4.25 pp (95% scene-bootstrap CI -0.25 to +8.75)
and Image Accuracy falls from 20.12% to 14.70%; this is not a broad gain or a
new evaluation. See the [artifact-level supplement](docs/pretrained_reuse/R2_NATIVE_BASELINE_LSTM_RECONCILIATION_20261029.md)
and the [original run reconciliation](docs/pretrained_reuse/R2_PATH_A_NATIVE_BASELINE_RECONCILIATION_20261028.md).

**2026-10-29 10:55 JST R0 checkpoint recheck:** Orbitune PID 22136 remained
live at log step 516,550 / 12,330,368,091 events; the latest health counters
reported zero non-finite loss/gradient events. Three recent log-rate samples
averaged 14,495 events/s. From 12,324,799,730 events, this projected the next
30M-event checkpoint in about 6 minutes; `model.pt` was saved at 10:54:28,
within that estimate. CPU-loaded checkpoint metadata confirms step 516,537 /
12,330,005,811 events, latest validation step 516,179 (loss -1.545699), and
best step 455,514 (loss -1.623135). The checkpoint SHA-256 is
`1e5dd30bff060155d8b311c18b4a7710bf755c145820b3f1f16a8b2f51d58201`.
RTX 3080 usage remained 15,821/16,384 MiB and C: had 385,966,080 bytes free.
No process or artifact was changed. This is a continuing `--steps 0` run, not
a completion ETA. See the [live checkpoint audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_1055JST.md).

**2026-10-29 10:35 JST R2-M RoboHEARD source screen:** The official challenge
page describes a real robot-centered multi-speaker AVSR task with multi-channel
audio, multi-view cameras, 100+ conversations, and speaker/environment-disjoint
30h/10h/20h train/dev/test partitions. Data are stated to be CC BY-NC-SA 4.0;
the development release is scheduled for 2026-10-30, and a Google Form
registration is required to receive data access by email; participant
dataset/model declaration is due by November 30. No form was submitted. Exact
group IDs, payload size, schema, consent details,
and both-input necessity are not yet verified; this is not yet a supplied-option
Decision task. Keep `REVIEW_NOT_ADMITTED`; no registration, agreement, data or
GPU work occurred. Three identical 40,771-byte metadata-page fetches averaged
0.174 s, which is not a dataset transfer estimate. C: free space remained about
0.372 GiB and the local GPU was occupied. See the [pinned observation and gate
analysis](docs/pretrained_reuse/R2_ROBOHEARD_SOURCE_SCREEN_20261029.md).

**2026-10-29 10:31 JST R2-M Humanola source screen:** Pinned public HF tree
metadata for `humanola-inc/egocentric-sample` reports 18.42 GB across 99 files
(largest single video 818.9 MB). The card reports nine synchronized video/hand-
pose/IMU episodes but no participant-grouped split; the publisher's catalog
describes per-episode consent without an exposed manifest tying it to this exact
revision. C: had about 0.37 GiB free; no payload or terms were accessed. Keep
`REVIEW_NOT_ADMITTED` for fitting. See
[`source screen`](docs/pretrained_reuse/R2_HUMANOLA_EGOCENTRIC_SAMPLE_SCREEN_20261029.md).

**2026-10-29 10:25 JST R2-M interface verification:** CPU-only focused tests
for the existing text-optional `ObservationBatch`, temporal/alignment contract,
masked fusion, and AV interaction pass: 53 passed. Mark the versioned observation
contract work item complete in the R2-M protocol. This verifies interfaces and
fallback masks only; it is not M1 modality quality, trained fusion evidence, or
the still-open both-input-required M2 task.

**2026-10-29 10:22 JST R2-M OpenMarcie metadata/code follow-up:** Read-only public
Kaggle JSON-LD reports CC BY-SA 4.0 and `requiresSubscription=true` for all
four archive parts, totaling 591,290,808,729 bytes (550.68 GiB). The paper's
subject-disjoint statement is for its 12-class HAR experiment; no pinned
participant-to-split manifest or equivalent guarantee for the other tasks was
located. The audio reported is instrumental, and paper results do not establish
that audio is necessary. C: had about 0.38 GiB free, so no archive or media was
downloaded and no terms were accepted. Keep OpenMarcie `REVIEW_NOT_ADMITTED`
pending rights/derivative-model policy review, split provenance, and feasible
storage. Three 12,295-byte metadata-page retrievals averaged 0.42 s; this is
not a data-transfer throughput estimate. See the dated
[source-screen follow-up](docs/pretrained_reuse/R2_OPENMARCIE_SOURCE_TASK_SCREEN_20261028.md#2026-10-29-metadata-and-split-follow-up).
The complete 22-blob official code tree at `fd72cf1ff11961520aafebc1f52587fd5bc2e74f`
contains no split-manifest path; its Apache-2.0 license applies to code only.

**2026-10-29 09:22 JST R0 live-state audit:** Orbitune PID 22136 remained live
at log step 513,650 / 12,249,743,497 events. Its latest saved checkpoint was
step 513,303 / 12,240,024,888 events, SHA-256
`928AB7CB9B07B8BEA8612A94A002758DBA45C34163C1CEEFCD10951F54524ABD`; recorded
validation best remains step 455,514. The run uses `--steps 0`, the RTX 3080
was 15,821/16,384 MiB used, and C: free space was 425,914,368 bytes. No process
or run was modified. PR #15 remains Draft/Open at its existing head. See the
[live-state audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0922JST.md).

**2026-10-29 R2-M AMI media preflight:** Exact HTTP `Content-Length` checks for
one submeeting each in the train (ES2002a), validation (ES2011a), and still
unexposed evaluation-candidate (ES2004a) groups total 98,332,121 bytes. A CPU
PyAV stream smoke decoded one train audio frame and one train video frame from
the official URLs without saving media or using the GPU: RV40 360×284 at 15
fps and SIPR 16-kHz mono. ES2002a video durations range 1,230.321–1,248.641 s;
the room-audio duration is 1,272.960 s, so alignment must be verified over the
common interval. ES2003a is unsuitable as the validation example because
Closeup1 returns 404; ES2011a has all four closeups. Annotation semantics are
still unresolved: ES2011a `speakersegAll.xml` includes channel 4, and its
browsable directory lacks the transcript HTML visible for ES2002a. The
ES2014a evaluation annotation was read in memory during schema inspection, so
quarantine the entire ES2014 group from any future locked evaluation; ES2004
is the only still-unexposed official scenario-evaluation group found here.
No model training or predictions occurred; the source stays
`REVIEW_NOT_ADMITTED` and C: free space was only 427,327,488 bytes. See the
[media preflight](docs/pretrained_reuse/R2_AMI_MEDIA_PREFLIGHT_20261029.md)
and [manifest](docs/pretrained_reuse/AMI_ES_MEDIA_PREFLIGHT_20261029.json).
This is subordinate to existing Draft PR #19 (head
`406f003a128a7991ec39dcb15608b381f003add4`), whose broader SA/SB preflight has
214 train and 111 validation candidate windows across five validation groups,
288 sampled PTS checks, and observed empty-room target-camera failures. Reuse
its external receipts. The same train-only qualitative visual-presence screen
covered 213 rows: 168 clearly framed faces, 21 ambiguous, and 24 with no clear
person; it did not verify speaking identity or timing. AMI remains blocked on
visible-speaker label validity and adequate independent validation, not on
another media-size probe.

**2026-10-29 R2-M AMI individual audio path follow-up:** The official ES2002a
listing includes individual `Headset-0..3` and `Lapel-0..3` low-bitrate tracks.
One `Lapel-0` SIPR frame decoded on CPU directly from the official URL; no full
track was saved. Official per-channel naming and the ES2002a roster map channel
0 to MEE006 / Closeup1; the publisher notes this participant's headset was
worn incorrectly but lapel is okay. Turn-label validity and temporal alignment
remain open, so this expands source availability only. AMI remains
`REVIEW_NOT_ADMITTED`; the label and independent-validation gates above remain
open.

**2026-10-29 R2-M AMI annotation-route follow-up:** The retained official
manual-annotation v1.6.2 archive contains the ES2011a development-validation
word/segment XMLs for all four agents (263 segments across channels 0–3).
Three training and four validation transcript intervals were checked against
the corresponding streamed Lapel tracks without saving media or running a
model. One validation channel-0 sample had lower RMS inside the annotation
interval than in the preceding second; keep it and audit the cohort rather than
filtering it. The archive hash and sample values are in the
[AMI preflight manifest](docs/pretrained_reuse/AMI_ES_MEDIA_PREFLIGHT_20261029.json).
This resolves the validation annotation-source lookup, not turn quality,
visible-speaker validity, synchronization, both-input necessity, or the small
number of independent groups. Evaluation labels remained unopened.

**2026-10-29 R2-M AMI development visibility reconciliation:** A retained
single-pass review was hash-checked and its 111 IDs exactly matched the pinned
PR #19 development candidate-window set; all 333 sampled frames and 12 contact
sheets validated. It found 69 clear target-camera faces, 6 limited, 2 without a
clear face, and 34 with no discernible person in the selected camera. This
review does not verify speaker identity, synchronization, or other camera
views. Keep every example; no filtering or model evaluation occurred. The
visibility and AV-quality gates remain open. Details and hashes are in the
[AMI preflight manifest](docs/pretrained_reuse/AMI_ES_MEDIA_PREFLIGHT_20261029.json).

**2026-10-29 09:47 JST R0 checkpoint confirmation:** Orbitune PID 22136
remained live at step 514,400 / 12,270,542,841 events. Its newly written
checkpoint at 09:45:28 was CPU-verified as step 514,381 / 12,270,020,976 events,
SHA-256 `b6a4f66cb3d2cfa3768e757369369682a5e8c1bc9b09d84eb01cf3317e78fa11`.
Latest saved validation is step 514,019 / loss -1.493842; the recorded best is
step 455,514 / loss -1.623135. `--steps 0` means the job has no finite end; this
checkpoint cadence is not a completion estimate. RTX use was 15,821/16,384 MiB
and C: free space was 0.39 GiB. No process or run was modified. See the
[09:47 live-state audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0947JST.md).

**2026-10-29 R2-M AMI candidate screen:** Official AMI metadata supports a
plausible synchronized active-speaker decision task and states CC BY 4.0 for
released signals/transcripts/some annotations. An official-metadata comparison
confirms no participant-ID overlap across the 15 ES scenario groups in the
scenario-only train/dev/evaluation partition; official annotation metadata
lists manual transcripts for all 60 ES submeetings. The official page reports
average per-meeting media sizes of 400 MB for low-size DivX or 40 MB RealMedia
video plus 2 MB audio; exact chosen-file sizes/checksums and RealMedia visual
suitability remain unknown, so no data were fetched and the source remains
`REVIEW_NOT_ADMITTED`. See the [source
screen](docs/pretrained_reuse/R2_AMI_ACTIVE_SPEAKER_SOURCE_SCREEN_20261029.md).

**2026-10-29 08:25 JST R0 checkpoint confirmation:** Read-only sampling
predicted the next save near 08:24:29; `model.pt` was written at 08:24:16.
CPU `weights_only=True, mmap=True` inspection verifies saved step `511837`,
`12,210,014,863` events, latest validation step `511477` / loss `-1.538318`,
and best step `455514` / loss `-1.623135`. The checkpoint SHA-256 is
`BEA851A2638D0383B173F23C1DB1D6F070399B902E6B13EFAF73503DDB3CAAC8`.
At 08:25 the process remained live; RTX use was 15,821/16,384 MiB and C: had
0.41 GiB free. The job still has `--steps 0`; this is checkpoint cadence, not
completion. No process, run, or checkpoint was changed. See the
[08:25 audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0825JST.md).

**2026-10-29 08:03 JST R0 progress recheck:** Orbitune PID 22136 remains live
at log step 511,150 / 12,190,945,719 events; the last saved checkpoint is
still step 510,759 from 07:49:52 (best step 455,514). The next 30M-event save
projects near 08:25 JST. RTX use is 15,821/16,384 MiB; C: free is 0.41 GiB.
No process or checkpoint was changed. See the
[live progress record](docs/pretrained_reuse/R0_LIVE_PROGRESS_20261029_0803JST.md).

**2026-10-29 07:51 JST R0 checkpoint update:** Orbitune PID 22136 remains live;
its 07:49:52 save was CPU-inspected as step 510,759 / 12,180,004,749 events,
with best step 455,514 and latest validation loss `-1.4775`. The checkpoint SHA
is `3534DF6D…A8A6EB5393C96`. RTX usage remains 15,821/16,384 MiB and C: free
space is 0.42 GiB, below the 2-GiB fit reserve. The next 30M-event save projects
around 08:24–08:25, not run completion. No process was changed and no Tiny Omni
fit was started. See the [R0 recheck](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0751JST.md).

**2026-10-29 07:19 JST R0 live checkpoint recheck:** Orbitune PID 22136 is
still running its epoch-sampled `--steps 0` job. Its latest verified periodic
checkpoint was written at 07:15:26 JST: step 509,683, 12,150,022,113 events,
SHA-256 `0BCE916A926FBF29BEEEC67A6799459B9F966ACA0F59DF2A4629EBBFEA93863D`.
The checkpoint stores no best step/loss, so it is a latest snapshot, not a
selected model. The 06:40:13→07:15:26 interval measured about 14,200 events/s; the
next 30M-event save projects near 07:50:39, not job completion. GPU was at
15,821/16,384 MiB and 99%; C: had 0.44 GiB free. No process or artifact was
changed and no Tiny Omni training was started. PR #15 remains Draft/Open at
this branch head and both Actions checks passed. See
[the live-state record](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0719JST.md).

**2026-10-29 07:08 JST R2 one-time unused-image transfer check:** The fixed
Path A readout selected at step 1,792 was scored on all 27,008 CLEVR-4
questions from 6,752 image groups not seen by that checkpoint, using cached
features on CPU. Accuracy was 19.65%, NLL 2.1633, Brier 0.8645, and ECE
0.0276, close to the saved same-checkpoint validation result (20.12% accuracy,
NLL 2.1651). That suggests no large extra penalty on unused image groups, but
does not identify the image-task bottleneck. This is a one-time result on the
original training partition, not independent validation or sealed-audit
evidence; no tuning or selection used it. It took 49.7 s; initial three-sample
throughput averaged 492.7 rows/s versus 543.5 overall. See
[the transfer report](docs/pretrained_reuse/R2_NATIVE_UNUSED_TRAIN_IMAGE_TRANSFER_20261029.md).

**2026-10-29 06:58 JST R2 CPU runtime replay:** The fitted Path A readout
was exported, reloaded, and run over all 24,346 pinned development-validation
rows using cached frozen features on CPU. It produced zero top-option
disagreements against the saved GPU score matrix; maximum absolute logit drift
was `3.53e-5` and macro metric deltas were under `1.3e-8`. The measured pass
took 58.8 s. The initial three 256-row throughput samples averaged 595.9
examples/s and underpredicted the remaining time by about 19 s as throughput
fell later. A first attempt's `2e-6` tolerance was too strict; the rerun used a
post-hoc `5e-5` tolerance, so this is evidence of CPU cached-feature execution
and close numerical agreement, not exact floating-point parity. It does not
cover raw encoders or change the product/quality gates. See the [full replay
record](docs/pretrained_reuse/R2_NATIVE_COMMON_CPU_REPLAY_20261029.md).

**2026-10-29 06:58 JST R0 process recheck:** Orbitune PID 22136 remained live;
the latest log was step 509,150 / 12,135,188,721 events, while its last
verified saved checkpoint remained step 508,603 at 06:40:13.621. RTX usage
was 15,821/16,384 MiB at 99%, and C: had about 0.72 GiB free. No GPU work was
started and the process/checkpoint were not modified. The process uses
`--steps 0`, so the next checkpoint cadence is not a completion ETA.

**2026-10-29 06:40 JST R0 checkpoint observed:** The next interval save was
written at 06:40:13.621 and CPU-loaded state verified step 508,603 /
12,120,026,000 events, 117,943,468 bytes, SHA-256
`2F33E7A6069FD84EA8A08C15B50FEBA9F94A77A458CD3C1D488C996C2BD5EC19`.
The 06:06:03→06:40:13 interval was 34m10s and 30,022,745 events. The run
remained live at 06:41:15 with RTX VRAM 15,821/16,384 MiB and C: free space
0.477 GiB. Its step limit remains `--steps 0`, so this is checkpoint cadence,
not a completion ETA. See the [R0 follow-up audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0640JST.md).

**2026-10-29 06:06 JST R0 checkpoint observed:** Orbitune's next interval
save appeared at 06:06:03.291, verified on CPU as step 507,526 /
12,090,003,255 events, 117,943,148 bytes, SHA-256
`4FF37521DD684C8437F7087260015365C2638E612C8D8EC12C6DAA35315FF1B`.
PID 22136 remained live at 06:06:49. The prior estimate was 87 seconds late;
measured checkpoint-to-checkpoint throughput projects another save near
06:40:40. This `--steps 0` run has no completion ETA, and its process/artifacts
were left untouched. See the [R0 live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0534JST.md).

**2026-10-29 R2-M Multimodal-Wireless metadata screen:** The official project
describes 161,400 synchronized simulation frames across 4 towns, 16 scenarios,
and 3 weather conditions, with camera, LiDAR, depth, radar, IMU/GPS, and CSI
data; its project page states CC BY 4.0. This could support a sensor/video
decision task. The associated paper defines the beam task and reports a
trajectory-disjoint 40/5/5 split, but no trajectory-ID manifest was located;
the official download page gives no payload size. Current C: free space is
about 0.48 GiB, so no dataset data were fetched. Keep `REVIEW_NOT_ADMITTED`
until payload size/checksums and the split can be reconstructed, then implement
the authors' 400-ms-history / 100-ms-horizon task before testing joint-input necessity
against unimodal and shuffled controls. The authors report 80.8% Top-1 with a
777M-parameter model, so this is a narrow research candidate, not evidence the
small product target is met. A metadata-only HEAD to the official OneDrive
share returned 403; no payload was requested. See the
[source screen](docs/pretrained_reuse/R2_MULTIMODAL_WIRELESS_SOURCE_SCREEN_20261029.md).

**2026-10-29 05:56 JST R0 checkpoint recheck:** Orbitune PID 22136 remains
live at step 507,200 / 12,080,946,174 events; its last saved checkpoint is
still step 506,449. The longer 05:47–05:56 log interval measured 30.6
steps/minute and 14,208 events/second, revising the next checkpoint estimate
to about 06:07:30 JST. This `--steps 0` run has no completion ETA. No process
or artifact was touched. See the [R0 live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0534JST.md).

**2026-10-29 05:49 JST R0 live recheck:** Orbitune PID 22136 is still present
at step 506,950 / 12,073,978,188 events; its last verified checkpoint is
506,449 at 05:31:28.203. Three samples measured 37.6 steps/minute; the latest
log reports 14,675 events/second, projecting the next 30M-event checkpoint
near 06:06 JST using the previously validated rate. This is not a completion
ETA (`--steps 0`). RTX VRAM remains 15,821/16,384 MiB and C: has 0.49 GiB free,
below the 2-GiB reserve. No process or artifact was modified. See the
[R0 live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0534JST.md).

**2026-10-29 R2-M XRFv2 Plus source screen:** This sensor/video action dataset
is relevant in content, but the linked Kaggle API says its data license is
`Unknown`; Zenodo's CC BY record attaches only the paper, while GitHub's MIT
license covers project code. Its payload is about 210.8 GB, and independent
group splits and both-input necessity remain unverified. No data was
downloaded or evaluated. Keep it unadmitted pending explicit data-rights
clarification. See the [source screen](docs/pretrained_reuse/R2_XRFV2_PLUS_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2-M MM-Motion source/rights screen:** Official Zenodo v2
metadata marks the synchronized video/IMU/plantar-pressure dataset open under
CC BY 4.0, but the publisher's agreement hosted at pinned upstream commit
`51a26dcf` requires signing, limits use to non-commercial research, prohibits
medical diagnosis and redistribution, and reserves access termination. Treat
that conflict as unresolved and keep MM-Motion unadmitted for product work.
The record contains 20 subject-named archives totaling 175.13 GB; its dataset
description names 33 subjects, but no split manifest or both-input necessity
was verified. Only metadata and the small agreement document were read; no
dataset payload, annotations, or media were downloaded. See the
[source screen](docs/pretrained_reuse/R2_MM_MOTION_SOURCE_RIGHTS_SCREEN_20261029.md).

**2026-10-29 05:34 JST R0 checkpoint timing verification:** Three read-only
log samples measured 31.75 steps/minute and 14,760 events/second. From the
12,043,312,981-event sample, this projected the next 30-million-event save at
about 05:31:28; `model.pt` was written at 05:31:28.203 with CPU-verified step
506,449 / 12,060,018,449 events. At 05:33:51 PID 22136 remained live and the
latest log was step 506,450. RTX 3080 use was 15,821/16,384 MiB and C: had
0.485 GiB free, still below the 2-GiB fit reserve. No Tiny Omni run or process
intervention occurred. The Orbitune command has no finite stop, so completion
has no ETA. See the [fresh read-only measurement](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0534JST.md).

**2026-10-29 04:57 JST R0 checkpoint recheck:** Read-only sampling of the
active Orbitune PID 22136 (started 2026-10-28 09:12 JST) predicted the next
30-million-event save near 04:57; the checkpoint was written at 04:57:10.
CPU `weights_only=True` inspection verifies step 505,374 and 12,030,012,963
events. `model.pt` is 117,942,316 bytes with SHA-256
`e363276109616051ae85438890d06fa035dd021664643f6e116f15990d7ebfcc`.
Three pre-save observations measured about 31.4 steps/minute and 14,689
events/second; the projected 13.5-minute save time was within seconds of the
observed interval, despite validation pauses. At 04:57:56 the process remained
alive and RTX 3080 use was 15,821/16,384 MiB. C: fell to 0.499 GiB free during
the save, below the existing 2-GiB fit reserve; do not start a Tiny Omni fit
until storage is safely restored and the cooperative GPU guard permits it.
No process or run was changed. See the [read-only measurement record](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0457JST.md).

**2026-10-29 R2 DROID later-shard transfer (PR #32, open draft):** Three
unchanged frozen readouts were replayed on 137 previously unopened official-
train episodes with zero overlap in episode, recording group, or sampled
frames; all 14 sites had appeared in the earlier cohort. Sensor-only scored
42.34% Accuracy / 1.4968 NLL; matched Video+sensor scored 39.42% / 1.7790,
with higher Brier and ECE as well. Preserve this as negative same-site,
group-disjoint development evidence; it gives no basis for another DROID fusion
fit or a broad video/sensor conclusion. No training, GPU, or sealed audit was
used. See the [frozen transfer report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-droid-temporal-transfer-20261028/docs/pretrained_reuse/R2_DROID_LATER_SHARD_FROZEN_TRANSFER_RESULTS_20261028.md)
and [PR #32](https://github.com/Unjuno/tiny-omni-decision/pull/32).

**2026-10-29 R0 live-state audit:** PR #15 remains Draft on the R0–R8 branch;
`main` has since moved to a separate ternary-first roadmap, and no merge was
made. The 2,048-update Path A baseline and 32,768-update long curve are already
complete, with step 8,192 selected by NLL; do not repeat them. The RTX 3080 is
occupied by Orbitune (15,821/16,384 MiB, 71%), and C: has 0.737 GiB free versus
the fit runner's 2-GiB output reserve. No process was interrupted and no new
fit was started. The current common-fit GPU guard also rejects the live
processes. See the [live audit and next-run gate](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0155JST.md).

**2026-10-29 04:32 JST R0 progress recheck:** Orbitune advanced to log step
504,600; the latest checkpoint remains step 504,302 from 04:23 JST. The
30-million-event checkpoint interval has about 21.64M events remaining; recent
versus since-checkpoint throughput projects another save in roughly 17–25
minutes, with evaluation pauses adding uncertainty. RTX 3080 remains at
15,821 MiB used and C: at 0.716 GiB
free. No Tiny Omni run was started. See the timestamped follow-up in the
[R0 live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0423JST.md).

**2026-10-29 04:23 JST R0 live recheck:** Orbitune PID 22136 is still training;
its CPU-only, `weights_only=True` checkpoint read verifies step 504,302 and
12,000,010,764 cumulative events. A validation record at the same step reports
loss -1.3606; `best_step` and `best_validation_loss` remain null. The run has no
finite step/epoch stop limit. The latest `model.pt` write is 04:23:02 JST; RTX
3080 remains at 15,821 MiB used / 100% utilization and C: has 0.717 GiB free.
Three one-minute log samples advanced 50 steps per 66–67-second interval
(about 45 steps/minute); evaluation pauses make wall-clock extrapolation
uncertain. No Tiny Omni job
was started because the local fit reserve is still unmet; the Orbitune run and
all artifacts were left untouched. See the [fresh read-only audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0423JST.md).

**2026-10-29 R2 GAViD source screen:** The publisher reports synchronized
video/audio group-affect clips, but its dataset terms restrict use to
non-commercial academic research; validation/test media also require a request
and approval. It is not admitted for product-directed work, and no payload or
media was downloaded. Its generated context and lack of demonstrated
both-input necessity would also require shortcut and unimodal controls in any
research-only evaluation. See the [source screen](docs/pretrained_reuse/R2_GAVID_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2 AV-SyncBench source screen:** Its pairwise temporal/semantic
audio-video synchronization tasks are structurally relevant, but the author
repository warns that original media rights remain with source creators; a
dataset-repository MIT tag does not resolve those rights. The full payload is
about 68.19 GB, and one included baseline uses non-commercial ImageBind. Keep
`REVIEW_NOT_ADMITTED`; current dataset-card terms were not independently
verified, and no payload was downloaded. See the [pinned source
screen](docs/pretrained_reuse/R2_AV_SYNCBENCH_SOURCE_SCREEN_20261029.md).

**2026-10-29 03:56 JST R0/R2 cross-branch recheck:** The current GPU process
is confirmed as Orbitune PID 22136 (parent 31624); 15,821 MiB VRAM is allocated
and C: has 0.72 GiB free, so no new fit was started. Open PR #36's DAVE screen
has a stronger both-required AV task design than AV-SyncBench but remains
blocked by non-commercial/agreement-bound component media. PR #35 keeps ZOD
unadmitted pending access, drive-level grouping, and measured fusion benefit;
PR #34 corrects GOOSE instance mapping without changing its unadmitted
disposition. See the [live cross-branch recheck](docs/pretrained_reuse/R0_R2_LIVE_CROSS_BRANCH_RECHECK_20261029_0356JST.md).

**2026-10-29 R2 HEARTBEAT 2027 screen:** This new challenge is the closest
task format found so far to typed Decision JSON: four-way audio turn-taking
and streaming AV event-response. However, dataset/media rights and consent are
not separately established, source-group independence is unverified, and its
AV dev shard is 9.28 GB against 0.72 GiB local free space. Keep
`REVIEW_NOT_ADMITTED`; no data was fetched. See the [source/task screen](docs/pretrained_reuse/R2_HEARTBEAT_2027_SOURCE_SCREEN_20261029.md).
The related Realtime-Venus paper describes scenario-planned/composed AV
construction, while the exact lineage and natural/composed fraction of these
challenge partitions remain unknown; do not treat this as an independent
natural-AV benchmark without a provenance audit.

**2026-10-29 R2 VideoFDB source screen:** VideoFDB is a close natural
audio/video interaction-evaluation match (237 dyadic clips, 11 dynamics, with
paired AV/audio-only baselines), but access is manually gated under NVIDIA's
non-commercial evaluation-only license. It has no train split and evaluates
rubric-scored conversational behavior rather than fixed-option probabilities;
no request, license acceptance, annotation access or media download occurred.
Keep `REVIEW_NOT_ADMITTED` for product training. See the [source/task/rights
screen](docs/pretrained_reuse/R2_VIDEOFDB_SOURCE_SCREEN_20261029.md).

The added Eidon CPU tests now skip cleanly when optional PyArrow/PyTorch are
absent; local Ruff/pytest/manifests pass and both CI runs for `0263aa97` are
green. No model run was repeated.

**2026-10-29 03:45 JST R0 recheck:** PR #15 remains Draft at `34b26cc0`, with
both checks green. Orbitune still occupies 15,821 MiB VRAM and C: has 0.73 GiB
free. A preserved Greatest Hits E0G0 receipt says `RUNNING`, but its matching
safety-stop record confirms an incomplete gradient-anomaly stop at step 5,385;
best step 4,096 was not reload-verified. Do not resume or repeat it. See the
[read-only reconciliation](docs/pretrained_reuse/R0_LIVE_RECHECK_20261029_0345JST.md).

**2026-10-29 02:11 JST R0 recheck:** Public GitHub API confirms PR #15 is still
open/Draft at `92a6ddaf500cf4499c513c5e48ce809fc0708aea` against
`09246d7b1821662a6b1eba47fc08c43f9dc3863c`; both push and PR checks for that
head are green. RTX 3080 remains occupied (15,821 MiB used, 355 MiB free), and
C: has 795,308,032 bytes free (0.741 GiB). Orbitune PIDs 22136/31624 and its
watchdogs remain present. No training or process intervention was performed.

**2026-10-29 02:45 JST R0 checkpoint recheck:** Orbitune remains active; its
live checkpoint was read on CPU and reports step 501,070 at 92.03% of epoch 1.
The command has `--steps 0` and no epoch stop limit, so no completion ETA is
available. GPU utilization dipped as low as 52%, but VRAM stayed at 15,821 MiB
used / 355 MiB free; C: has 0.737 GiB free. The latest `model.pt` timestamp is
02:39 JST. Do not treat utilization dips as an idle GPU or attempt a fit while
the 2-GiB output reserve also fails. PR #15 commit `7dba9b5e` passed its push
and PR checks. Details and exact read-only inspection method are in the
[follow-up audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0155JST.md).

**2026-10-29 03:16 JST R0 follow-up:** Orbitune remains active at log step
502,200. Its latest `model.pt` was saved at 03:13:31 JST and SHA-256 verified;
the checkpoint step is not inferred from the later log step. Five 50-step log
intervals averaged about 94 seconds, while the next scheduled 30-million-event
checkpoint is estimated at roughly 34 minutes from the last save. RTX 3080
remains fully occupied and C: has 0.735 GiB free, below the 2-GiB fit reserve.
No process was interrupted or manually changed; the live run wrote its normal
checkpoint, and no new fit was started. See the
[live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_0155JST.md).

**2026-10-29 R1/R7 parameter accounting:** The common Path A fit's exact
pre-quantization count is 196,254,874 (195,365,529 frozen encoders plus an
889,345-parameter readout). The v4 modular package's 170,499,716 unique count
comes from a different task-scoped encoder inventory and is not the same
trained model. Its 687,254,772 bytes still exceed the provisional 200 MB
package target. The pinned checkpoint files and both receipts were re-hashed.
See the [parameter accounting reconciliation](docs/pretrained_reuse/R1_R7_PATH_A_PARAMETER_ACCOUNTING_20261029.md).

**2026-10-29 R2-M audio/sensor source screen:** The Zenodo ADL collection has
25 activity labels, 14 participants, synchronized ambient audio and multiple
environmental sensors, but it contains no video and provides no declared
participant-disjoint split in the public metadata inspected. The paper states
CC BY 4.0; the Zenodo record's rendered rights field is blank, so payload terms
still need confirmation. Its 4.6 GB archive is not downloaded: C: remains below
the 2-GiB reserve. Keep REVIEW-only for a narrow audio/sensor candidate; it
cannot satisfy the video requirement or establish both-input necessity without
participant IDs and matched unimodal controls. See the [source screen](docs/pretrained_reuse/R2_ADL_AUDIO_SENSOR_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2-M EgoInertia-MI candidate screen:** The official project page
describes synchronized egocentric video and three wearable IMUs, with action
recognition and simulated motor-impairment severity tasks and a subject-disjoint
five-fold protocol. This is unusually well aligned with the optional-sensor
path, but the data are access-restricted through Zenodo and the page does not
name an explicit data license. No request, payload access, or training was
performed. Keep REVIEW-only pending access, payload license, privacy/consent,
and exact split/payload verification. It is a non-clinical simulated benchmark,
not a clinical decision dataset. See the [source screen](docs/pretrained_reuse/R2_EGOINERTIA_MI_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2-M WEAR screen:** WEAR has synchronized egocentric video and
wearable inertial data from 22 participants across 18 outdoor activities, but
the official dataset license is CC BY-NC-SA 4.0 and its raw payload is about
130 GB. Do not use it for product-directed training or acquire it under the
current storage constraints. Preserve as a research-only reference; no payload
or media was downloaded. See the [screen](docs/pretrained_reuse/R2_WEAR_VIDEO_IMU_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2-M Luel IMU Egocentric screen:** The card describes paired
Video/30-Hz IMU and structured work-task steps, but marks the data All Rights
Reserved, forbids use without permission, and gates the 250,000-hour corpus.
The 10-clip preview has no explicit separate permission or independent split.
Do not fetch or use the preview pending explicit payload rights. No access
request or download was made. See the [screen](docs/pretrained_reuse/R2_LUEL_IMU_EGOCENTRIC_SOURCE_SCREEN_20261029.md).

**2026-10-29 MEVA license follow-up:** Official MEVA terms explicitly license
the dataset under CC BY 4.0, and the WACV paper reports IRB oversight and actor
consent. This resolves the prior dataset-license uncertainty for a bounded
video-only probe, subject to privacy/personality rights and attribution. It is
not natural AV or IMU fusion evidence. The partial annotation checkout remains
incomplete, no video media was obtained, and no task/split or storage/runtime
gate is yet satisfied. See the [license recheck and remaining gates](docs/pretrained_reuse/R2_MEVA_GIT_METADATA_ACQUISITION_20261028.md).

**2026-10-29 MEVA training-metadata preflight:** At the pinned annotation
revision, the complete `kitware-meva-training` subtree has 1,443 clips and
35,174 labels across 37 classes. Exact time-window/site grouping yields 741
groups; conservative date/site grouping only 28, and participant independence
across clips remains unverified. The rarest labels occupy only 1, 2 and 3
date/site groups, so a disjoint three-way split cannot cover all 37 labels in
every split. No evaluation-level annotation contents or video media were
accessed. Keep as a video-only research candidate pending a principled
narrower task, independent split, and storage/runtime gates.
See the [pinned preflight and counts](docs/pretrained_reuse/R2_MEVA_GIT_METADATA_ACQUISITION_20261028.md).

**2026-10-28 R2 FusionX V3 transfer/source screen:** Three small RGB files
from one pinned task were downloaded from TouchTronix's CC BY 4.0 V3 release
and hash-verified for a source-host transfer probe. Their weighted rate was
0.150 MiB/s; linear extrapolation to the published 15.9 GB is roughly 28–30
hours, with substantial uncertainty. The complete payload exceeds available
disk. The release exposes only one `train` split and 11 task variants; paired-
input necessity, independent task/actor validation, and participant/privacy
terms remain unresolved. One inspected frame contains a private-looking
workstation and computer display. Keep `REVIEW`; no training or quality
evaluation. See the [source and transfer screen](docs/pretrained_reuse/R2_FUSIONX_V3_SOURCE_TRANSFER_SCREEN_20261028.md).

**2026-10-28 R2-M synchronized video+sensor source screen:** C-MHAD is a
task-aligned continuous video+IMU candidate (12 participants, two action
families), but the official download page does not state dataset-payload
licensing or participant-media terms. The paper's CC BY license is not treated
as a license for separate data files. No media was fetched and no access request
was sent. Keep REVIEW-only pending rights, privacy, and immutable-version
checks; its 12 independent participants would support only a narrow diagnostic.
See the [source screen](docs/pretrained_reuse/R2_C_MHAD_VIDEO_IMU_SOURCE_SCREEN_20261028.md).

**2026-10-28 R1/R2 open-PR reconciliation:** PR #16's pinned PE-Core T16
CPU/runtime and CLEVR-4 readout evidence does not outperform the existing
Path A count result and has a non-commercial implementation-license constraint.
PRs #31/#32 add a DROID shard/site audit and fixed-readout transfer: the sample
contains no new collection site, and VideoPrism+sensor remains below
sensor-only on matched later-shard records. A post-hoc 14-site bootstrap of
the saved predictions gives joint-minus-sensor Accuracy −2.92 pp (95% CI
−8.08 to +0.97), NLL +0.282 (+0.160 to +0.487), and Brier +0.0887 (+0.0448 to
+0.1595). No rerun or product promotion is justified. PRs #31/#32 remain open
and unchanged. See the [cross-branch evidence
reconciliation](docs/pretrained_reuse/R1_R2_CROSS_BRANCH_RECONCILIATION_20261028.md)
and [reproducible bootstrap audit](https://github.com/Unjuno/tiny-omni-decision/blob/b8245103d221daea225f9666862f6b149462aa1b/docs/pretrained_reuse/R2_DROID_LATER_SHARD_PAIRED_SITE_BOOTSTRAP_20261028.md).

**2026-10-28 R2 Path A baseline-run reconciliation:** The common frozen-feature
four-modality Decision readout already completed a 2,048-update seed-17 fit
and matched LSTM control. Fixed-endpoint macro Accuracy was 30.58% / 30.71%;
macro NLL was 1.6910 / 1.7071. The LSTM best checkpoint's higher Video score
(35.75%) coincided with lower Image and macro Accuracy, so it was not adopted.
This closes the baseline-fit work package, not the task-quality, joint-input,
or product gates. No repeat training is warranted. See the [run reconciliation
and hashes](docs/pretrained_reuse/R2_PATH_A_NATIVE_BASELINE_RECONCILIATION_20261028.md).

**2026-10-28 R2 AV source gate remains open:** Official QIVD terms bind on
download and restrict results to internal non-profit research, prohibit
commercial use and third-party data combination. Tsinghua AVQA has relevant
`Both` multiple-choice labels but personal/classroom/non-profit use terms and
YouTube-sourced clips. Neither is admitted for the multi-source product model;
no data or agreement was accessed. See the [source/license screen](docs/pretrained_reuse/R2_AV_SOURCE_LICENSE_TASK_SCREEN_20261028.md).

**2026-10-28 R2 CONVERGE source screen:** CONVERGE offers synchronized RGB-D
and radio inputs with a three-state blockage target, but its official README
specifies CC BY-NC-ND 4.0, the rendered Zenodo License field is blank, and the
4.8 GB train/validation archives leave only about 2.3 GiB on C: before
extraction. No archive was downloaded or used. Keep it out of product-directed
training pending permissive rights and an independent test protocol. See the
[source screen](docs/pretrained_reuse/R2_CONVERGE_SOURCE_TASK_SCREEN_20261028.md).

**2026-10-28 R2 OpenMarcie follow-up candidate:** OpenMarcie is the closest
new natural synchronized video/audio/wearable-sensor source found in this
screen, with 36 participants and activity/alignment tasks. Its official
materials reviewed here do not state the dataset license or participant-level
split protocol; the code's Apache-2.0 license is not data permission. No payload
was accessed. Prioritize rights, archive identity, and group-split verification
before any download or training. See the [source
screen](docs/pretrained_reuse/R2_OPENMARCIE_SOURCE_TASK_SCREEN_20261028.md).

**2026-10-28 R2 LATAM Egocentric + IMU screen:** The card reports CC BY 4.0
and research-use participant consent, but only 20 clips / 39 minutes, one
unpartitioned split, and 3.41 GB total payload. Only 13 clips have recoverable
clip-level synchronization; 7 session-level IMU streams have unknown offsets.
No payload was downloaded. This is at most a small input/sync prototype after
storage, participant-group and consent review; it cannot establish two-input
necessity or R2-M generalization. See the
[source screen](docs/pretrained_reuse/R2_LATAM_EGOCENTRIC_IMU_SOURCE_SCREEN_20261028.md).

**2026-10-28 R2 open-source PR reconciliation:** PR #35's ZOD schema ties 2D
and 3D boxes by annotation UUID but leaves access and split/overlap gates open.
PR #36 identifies a FZI-AURA access-control discrepancy: its previously
retrieved sample is quarantined and must not be read or used. DAVE has a
promising both-required AV diagnostic but combines non-commercial and
agreement-bound sources. Neither PR admits new model work; both remain open
Drafts. See [the read-only reconciliation](docs/pretrained_reuse/R2_OPEN_PR_SOURCE_RECONCILIATION_20261028.md).

**2026-10-28 R2 MUVS candidate screen:** MUVS offers synchronized sports
video + 18 phone-sensor channels, 5,039 three-second fragments, and expert
camera-selection labels, a promising supplied-option task shape. Its Zenodo
license field and dataset-page consent terms are unstated; event split
independence and actual paired-input necessity are unverified. The full video
payload is reported as ~164.9 GB. No archive or media was downloaded; keep
`REVIEW` pending rights, event-group, and selected-payload-size gates. See the
[source/task screen](docs/pretrained_reuse/R2_MUVS_SOURCE_TASK_SCREEN_20261028.md).
The 2026-10-29 Zenodo page recheck still shows no rights/license and now shows
a 1.1 GB record-level data-volume statistic beside its 9.7 MB listed metadata
archive; these do not resolve the publisher's ~164.9 GB external video payload.

**2026-10-29 R2-M EgoViz-120 metadata-only screen:** The publisher states
CC BY 4.0 and describes synchronized video/IMU/action labels; a hash-pinned
871 KB index contains 5,392 clips across 180 session groups. Labels are
model-generated, no official split is stated, participant independence across
groups is unverified, and the dataset is 7.25 TB. The preview MP4 overlays labels;
raw video requires large MCAP files. Keep `REVIEW`/not admitted; metadata only
was fetched, no media or audit data was opened. See the
[source screen](docs/pretrained_reuse/R2_EGOVIZ120_SOURCE_SCREEN_20261029.md)
and [machine-readable metadata](manifests/pretrained_reuse/egoviz120-metadata-screen-20261029.json).
**2026-10-29 EgoViz-120 provenance recheck:** The publisher's dataset-specific
card still says action labels are model-generated and not human-verified, while
its general company site describes human review and random re-audits. Treat the
dataset-specific label provenance as unresolved pending a dataset-specific
clarification or sample audit. The current GitHub license at
`humaidtech/EgoViz-120@9c91b23a66fcaa1fee083b679d5993fdfff9a64e` explicitly
states CC BY 4.0 including commercial use with attribution; the exact Hugging
Face card revision is `7a84ecd3411b9f219b02f273c438f1ea12d74f52`. This resolves
the publisher-license-text check, not label quality, participant independence,
both-input necessity, or the local media-storage gate. No media was downloaded.
See the [pinned provenance recheck](docs/pretrained_reuse/R2_EGOVIZ120_PROVENANCE_RECHECK_20261029.md).

**2026-10-28 R2 GRID Path A readouts completed:** Existing pinned frozen
audio/video features were evaluated on two fixed development tasks. The
six-slot joint readout (88.35%) did not improve over audio-only (89.19%); a
video-swap control matched it, and video-only was 20.57%. A separate
four-choice clip-matching readout reached 31.25%; a four-frame temporal offset
scored slightly higher. This closes this GRID candidate for quality iteration,
not R2-M or the product goal. No GPU run or repeat download was needed. See
[the measured results and artifact hashes](docs/pretrained_reuse/R2_GRID_PATH_A_READOUTS_20261028.md).

**2026-10-28 R0/R2 cross-branch recheck:** Read-only review of current PRs
#31/#32 and the latest DROID, ADL, GOOSE and GRID branches found no new
admitted task that justifies another fit or large download. DROID's later-shard
sample found no new site among six sampled shards; its four-site VideoPrism +
sensor readout remained weaker than sensor-only. ADL's 25 archives are fully
authenticated, but its fixed split has only 12 train and 2 validation
participants, one uncovered class, and no video. GOOSE's train-only image/point
cloud instance keys agree inconsistently. GRID now has a corrected direct-source
speaker split and a fixed AV correspondence probe, but the probe is weak and
poorly calibrated on eight development speakers; cross-corpus speaker identity
and product-use rights remain unresolved. R2-M and Path A joint-quality gates
remain open. See [the evidence reconciliation](docs/pretrained_reuse/R0_R2_CROSS_BRANCH_RECHECK_20261028.md).

**2026-10-26 R2 UnAV-100 source preflight:** Its official split CSV contains
10,790 unique YouTube IDs (6,489 train / 2,134 validation / 2,167 test) with no
cross-split ID overlap and positive spans. This is a promising natural AV event
task, but the project site says CC BY 4.0 while the pinned Hugging Face card
says CC BY-NC-SA 4.0; underlying video copyrights remain with source owners.
An official repository collaborator's Google Drive folder was confirmed
read-only to list annotations and VGGish/I3D feature archives, but no artifact
payload was fetched and the folder metadata does not resolve component terms.
Keep it REVIEW-only until the license discrepancy and source-video rights are
resolved.
[CSV hash and scope](docs/pretrained_reuse/R2_UNAV100_SOURCE_SCREEN_20261026.md).

**2026-10-26 R2 additional AV source screen:** MCIF's pinned CC-BY-4.0
release has only test configurations, all sharing the same underlying audio and
video, and its QA references are not supplied-option Decision labels. SlideSpeech
is contextual AVSR; the publisher licenses metadata, while the corpus paper
limits use to academic research and the source media are YouTube recordings.
Neither source currently admits a supervised Path A AV Decision fit. No payload
or media was downloaded/opened. [Source/task screen](docs/pretrained_reuse/R2_MCIF_SLIDESPEECH_SOURCE_SCREEN_20261026.md).

**2026-10-26 R2 Perception source-candidate validation transfer complete:** The
fixed epoch-7 readout scored 138 previously integrity-checked, but non-blind,
official-validation events once: joint audio/video 52.17% set Accuracy / 1.4637
NLL. Video-only was 55.07%, audio-only 13.77%; mismatched audio reduced
Accuracy to 41.30%. A train-label option/none prior reached 57.25% by predicting
`none_visible` for every event (0% visible-source hit), showing overall
Accuracy is misleading here. No fitting or checkpoint selection used these
IDs; this closes only the pending extraction/evaluation step, not R2-M or the
broader multimodal Decision gate. Keep this readout diagnostic-only and do not
fit again on the exposed cohort. See the [protocol, controls, intervals and
hashes](docs/pretrained_reuse/R2_PERCEPTION_SOURCE_CANDIDATE_VALIDATION_TRANSFER_20261026.md).

**2026-10-26 R2-M synthetic temporal architecture evidence reconciled:** The
independent `codex/r2-synthetic-av-gated-temporal-20261026` branch completed a
16,384-update residual-gated LSTM run on the existing frozen synthetic AV task.
It matched the plain raw-input LSTM at 100% selected validation Accuracy but
had slightly worse NLL/ECE; the earlier conditional gates stayed at chance.
This supports no gate benefit and is synthetic-mechanics evidence only, not
natural AV transfer. An artifact audit found the residual-gate report copied
the earlier bypass run's prediction SHA; the residual run's evaluation receipt
and 71,680-row predictions agree on SHA `66ff5c760e556a29b05ea366f7f4392bb48e6505a7aee61076949d2b0d2e2d83`.
The original branch and artifacts remain untouched. Do not repeat this synthetic
fit; the next R2-M gate is a naturally recorded, rights-cleared task with
independent source groups. See the [cross-branch reconciliation and hashes](docs/pretrained_reuse/R2_SYNTHETIC_AV_CROSS_BRANCH_RECONCILIATION_20261026.md).

**2026-10-26 R1 frozen component runtime recheck:** A fresh CPU-only process
reloaded the pinned VideoPrism-B, Ruri v3-30m, and EfficientAT DymN04-AS
checkpoints and produced finite expected-shape outputs; checkpoint hashes and
run receipts were rechecked locally. The combined towers contain 153,031,959
parameters and 612,390,421 checkpoint bytes, before task readouts and
preprocessing. This supports runtime feasibility only: generated text/tone
inputs do not measure task quality. The provisional parameter budget remains
plausible for this subset, while raw checkpoint size already exceeds 200 MB.
See the [R1 CPU recheck and receipts](docs/pretrained_reuse/R1_FROZEN_COMPONENT_CPU_RECHECK_20261026.md).

**2026-10-26 R2 input-grounding and fusion recheck:** A fixed 168-example
Open Images probe reached 79.17% Accuracy, but its selected scorer produced
identical top-1 decisions after image features were zeroed or deranged. Treat
that result as a candidate/text-prior diagnostic, not image-grounded quality.
A DROID video+robot-state next-action readout reached 36.5% validation
Accuracy versus 44.0% for sensor-only; paired episode intervals also favored
sensor-only. Do not continue fitting either exposed task. The latest VisTouch
Git recheck confirms the repository contains no native synchronized media.
These findings sharpen the R2-M admission gate: before fitting, measure
option-/metadata-only predictability and verify the target requires the
proposed observation streams, then retain matched unimodal and perturbation
controls. See the [Open Images controls](https://github.com/Unjuno/tiny-omni-decision/blob/54a1505b71a5392ad9c43d78817592cc67cc09df/docs/pretrained_reuse/R2_OPENIMAGES_INPUT_CONTROLS_20261026.md),
[DROID video+sensor results](https://github.com/Unjuno/tiny-omni-decision/blob/54a1505b71a5392ad9c43d78817592cc67cc09df/docs/pretrained_reuse/R2_DROID_NEXT_ACTION_VIDEO_SENSOR_20261026.md),
and [VisTouch Git recheck](docs/pretrained_reuse/R2_VISTOUCH_GIT_RECHECK_20261026.md).

**2026-10-26 R2 natural-image Path A follow-up:** An independent branch's
observer-disjoint eight-species iNaturalist CC0 probe selected 81.13% macro
Accuracy / 0.5584 macro NLL with frozen 114.37M-parameter VideoPrism-B plus a
6,152-parameter readout. Zeroed and permuted images fell near chance, but the
live-API cohort has no sealed audit, train-side spot checks found weakly grounded
examples, and the 2,048-epoch curve overfit after epoch 1,276. This is narrow
closed-set image evidence, not general option-conditioned Decision quality.
No further fit on those exposed IDs. The full protocol and artifact hashes are
in the [iNaturalist report](https://github.com/Unjuno/tiny-omni-decision/blob/19392c947da150b4f0e4e1aca4cccef5374c1bdf/docs/pretrained_reuse/R2_INATURALIST_VIDEOPRISM_READOUT_20261026.md).

**2026-10-26 R2 AV-Actions source gate:** The official Audio-Visually Indicated
Actions dataset has synchronized audio, video and acoustic images, but only
nine people/three scenarios, request-only non-commercial research access, and a
published split generator that can place chunks from one recording across
splits. No request, data acquisition or fitting was performed. Keep it
REVIEW-only pending authorized access, a grouped split and an audio-vs-video
task-dependence preflight. [Source and split screen](docs/pretrained_reuse/R2_AV_ACTIONS_SOURCE_SCREEN_20261026.md).

**2026-10-26 BridgeData source gate:** BridgeData V2 remains a useful
video+robot-state task-design lead, but the official TFDS release is 387.49
GiB and the reviewed raw converter randomly partitions trajectories within
dated collection folders. Group independence of the published TFDS train/test
splits has not been verified. Keep it `REVIEW`; do not acquire shards or fit a
readout until the actual episode-group split and a task requiring both image
and state inputs are established. See the [source and split screen](https://github.com/Unjuno/tiny-omni-decision/blob/2583f3a76738b6c5772741d47724518f2b616dae/docs/pretrained_reuse/R2_SENSOR_VIDEO_SOURCE_SCREEN_20261026.md).

**2026-10-25 Perception frozen-readout validation transfer complete:** The
fixed 200-event official-validation cohort was scored once using frozen
Audio/Video/Joint readouts after 200/200 VideoPrism and EfficientAT feature
extraction. The joint head reached 66.50% Accuracy (Audio 57.00%, Video
58.00%); grouped-bootstrap evidence supports an Accuracy increase over either
single-modality head, while calibration improvements are uncertain. This
label-balanced, source-screened visibility probe is nonblind and cannot close
the joint-quality gate. See the
[`transfer diagnostic status`](docs/pretrained_reuse/R2_PERCEPTION_VALIDATION_TRANSFER_20261025.md).

**2026-10-26 R2 source-candidate validation integrity:** Before reusing the
existing epoch-7 source-candidate readout, a CPU audit verified the frozen
official-validation cohort, zero video-group overlap with its 320-train/80-dev
fit, all 276 candidate MP4/WAV hashes, and the checkpoint hash. It found 138
eligible events across 138 video groups. This cohort was previously inspected
and visibility-balanced, so it is a non-blind development-transfer diagnostic;
no candidate-task metrics have been computed. The extraction path currently
supports only train/development. Add and test an explicit official-validation
path before a one-time fixed-checkpoint evaluation; do not tune on these IDs.
See the [integrity report](docs/pretrained_reuse/R2_PERCEPTION_SOURCE_CANDIDATE_VALIDATION_INTEGRITY_20261026.md).

**2026-10-25 R2 one-time CoGenT-B audit complete:** The frozen ViLT
image+question option readout scored 57.90% overall Accuracy / 0.7227 NLL and
59.05% macro-by-answer-kind Accuracy / 0.7740 NLL on 15,000 held-out valB
images. Real-image inputs exceeded gray and same-split shuffled controls under
paired 10,000-replicate image bootstrap, but accuracy varied from 48.25% on
Boolean to 74.59% on Count. This is OOD transfer evidence for one research-only
image+question probe, not general Omni or product quality; CoGenT-B is closed
to further tuning. [Protocol, detailed scores, hashes and limits](docs/pretrained_reuse/R2_CLEVR_COGENT_B_ONE_TIME_AUDIT_20261025.md).

**2026-10-25 R2 CLEVR option-readout/data implementation:** Added a
deterministic image-grouped subset builder, frozen-feature 28-answer readout,
supplied-option loss, per-kind/program metrics, and separate best/terminal
readout saves. The pinned official annotation archive produced 16,384 train
and 4,096 official-validation examples with zero image and normalized-question
overlap (attempt-03). Two selected media files passed range/ZIP CRC checks, and
a CPU ViLT forward produced `[2,768]` frozen features; this remains an
input-path smoke. Manifest hashes and answer/question-type counts are recorded in the
[`CLEVR compositional preflight`](docs/pretrained_reuse/R2_CLEVR_COMPOSITIONAL_OPTION_PREFLIGHT_20261025.md).
CPU tests pass (2,418 passed, 11 skipped), Ruff passes; the corrected push is
green on GitHub Actions. Fixed gray-image and split-local shuffled-image
controls pass CPU feature-extraction smoke. All 20,480 selected media images
have now been downloaded from the pinned archive; a full CPU audit matched the
receipt SHA-256 for every file, verified both frozen manifest hashes, decoded
all images at 480x320, and found no missing or extra files. Acquisition took
7,934.7 seconds. No full feature cache or quality score exists yet. GPU feature
extraction/fitting waits on the existing RTX 3080 process and an older queue
that must run first. CoGenT-B remains unopened.

**2026-10-25 R2 CLEVR compositional option-task preflight:** The official
CLEVR v1.0 CC BY 4.0 annotation archive is pinned locally; the 19.02 GB image
archive was inspected through ZIP metadata-only HTTP ranges. The candidate
asks image + question → probabilities over supplied, same-answer-kind choices,
using all 28 CLEVR answers rather than the earlier fixed CLEVR-4 taxonomies.
A pinned 117.59M ViLT VQAv2 checkpoint passed one CPU forward on a real
official-train image/question, but no quality score or fitting has run. Its
closed VQA head lacks `cyan`, `cube`, and `sphere`, so the candidate replaces
that head with a 28-answer readout. An exploratory internal train reserve was
opened while counting labels and is explicitly disqualified as a blind audit;
CoGenT-B validation remains unopened for a prospective one-time project-held-
out audit. The 28-answer mapping / supplied-option softmax now has CPU tests
(4 passed; Ruff clean). The selected train/validation media cache is now fully
acquired and hash/CRC/decode verified; no model checkpoint or quality score
exists yet. The RTX 3080 remains occupied by the existing Teacher run, so no
GPU feature extraction or fit has started. Details and hashes:
[`CLEVR compositional option preflight`](docs/pretrained_reuse/R2_CLEVR_COMPOSITIONAL_OPTION_PREFLIGHT_20261025.md).

**2026-10-25 R2 ViLT long-question input fix:** A tokenizer audit found 75/16,384
frozen train questions and 14/4,096 validation questions exceed ViLT's pinned
40-position limit. The feature extractor now preserves every token by splitting
only long questions into overlapping 40-token windows (8 content-token overlap)
and averaging per-window pooled features; short-question inputs retain direct
forward parity. CPU tests and the full pinned-checkpoint input smoke pass, but
no feature cache or quality score has been produced. Full CPU suite: 2,426
passed, 11 skipped; Ruff passes. GPU work remains queued behind the existing
Teacher run and previously queued full-extent job. Details and hashes are in
the [CLEVR preflight report](docs/pretrained_reuse/R2_CLEVR_COMPOSITIONAL_OPTION_PREFLIGHT_20261025.md).

**2026-10-25 R2 CLEVR-4 full-validation transfer reconciliation:** Recomputed
metrics from the saved, one-shot VideoPrism-B fixed-readout predictions for all
19,988 official 100k validation images; all 79,952 decisions were valid and
per-taxonomy Accuracy/NLL/Brier/ECE matched the report. Macro Accuracy is
87.41%, with Count at 60.44%; the runtime's 114.5M resident parameters fit the
provisional parameter range, but its 458 MB payload exceeds the provisional
200 MB package target. This is a fixed synthetic taxonomy
transfer result; the validation is exposed, so it is closed to tuning. Local
metadata says the run completed and its prediction SHA matches, although the
source report's opening status sentence still says the replay is running. See
the [task battery update](docs/pretrained_reuse/R0_TASK_BATTERY_DRAFT_20261021.md)
and [transfer report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/clevr4-100k-transfer-eval-20261023/docs/pretrained_reuse/R2_CLEVR4_100K_TRANSFER_EVAL_20261023.md).

**2026-10-25 07:26 JST R0 live recheck:** The continuation worktree is clean
at `f754d28`; all three public GitHub checks for that commit are green. The
existing Teacher process (PID 26920, supervisor 15928) remains active and owns
16,122/16,384 MiB on the RTX 3080 Laptop. Its history reaches step 1280, while
the persisted trainer state and best checkpoint remain at 1152; the exact
operation after the latest history record is unknown. The separately queued
full-extent × fresh-coverage run (worker PID 33712) is still waiting for an idle
GPU and has created no outputs. No process or artifact was changed, and no
factorial estimate is available. See the
[`07:26 JST live audit`](docs/pretrained_reuse/R0_LIVE_AUDIT_20261025_0726JST.md).

**2026-10-25 07:46 JST R0 live recheck:** PR #15 remains Draft/Open and both
remote research refs are at `82b3d8c`; all three CI runs for that commit are
green. The Teacher process is still alive and accumulated about 11 CPU seconds
over a 12-second observation interval while GPU utilization varied 0–99%; its
last history/state artifacts remain step 1280/1152, respectively. The HotGPU
queue remains `WAITING_FOR_IDLE_GPU`, with no experiment outputs created. No
process or artifact was modified. See the
[`07:46 JST live audit`](docs/pretrained_reuse/R0_LIVE_AUDIT_20261025_0746JST.md).

**2026-10-25 R2 cross-workstream reconciliation:** Recent open PRs add two
relevant results and more source gates. The matched CLEVRER/generated-tone AV
diagnostic selected a gated temporal LSTM at 83.51% Accuracy / 0.3534 NLL on
1,110 task-specific validation rows, versus 75.14% for matched audio-only; the
paired scene interval for the Accuracy difference was +5.89 to +10.83 points.
This is a synthetic timing task with cross-task media reuse and exposed
development validation; video-only remained at chance, so it does not close
natural-AV or four-modality quality. A separate TRAIN-only perturbation shows
the selected LSTM responds to both streams and order, but is not generalization
evidence. The CSQA2 frozen MiniLM transfer probe reached 51.79% on its official
development split, equal to majority-class accuracy, with 93.5% of predictions
on “no”; do not treat this as useful general text quality. New AVSBench,
ASAPS, KilometerAudio, EvaOmni and AVI-Bench screens admit no new product
training source because of rights/access or split/task gaps. Details: [synthetic
AV comparison PR #21](https://github.com/Unjuno/tiny-omni-decision/pull/21),
[TRAIN dependency PR #26](https://github.com/Unjuno/tiny-omni-decision/pull/26),
[source reconciliation PR #27](https://github.com/Unjuno/tiny-omni-decision/pull/27),
[CSQA2 text probe PR #29](https://github.com/Unjuno/tiny-omni-decision/pull/29),
and [AVSBench/ASAPS source gate PR #30](https://github.com/Unjuno/tiny-omni-decision/pull/30).

**2026-10-25 R2 additional source-gate reconciliation:** Recent independent
metadata-only screens do not admit AVRT-20K, RoboPro, or otoSpeech as training
or quality-evaluation sources. AVRT-20K has one published split and unresolved
media rights; RoboPro exposes only a train split and continuous robot commands,
not the frozen option-probability target; otoSpeech has unresolved participant
and stimulus permissions plus only 18 sessions / ten reported scene pairs, so
group-independent evaluation is not established. No labels or media were
accessed. See the [AVRT-20K screen](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-csqa2-text-decision-20261024/docs/pretrained_reuse/R2_AVRT20K_SOURCE_GATE_20261024.md),
[RoboPro screen](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-robopro-source-screen-20261024/docs/pretrained_reuse/R2_ROBOPRO_SOURCE_SCREEN_20261024.md), and
[otoSpeech screen](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-robopro-source-screen-20261024/docs/pretrained_reuse/R2_OTOSPEECH_TASK_SOURCE_SCREEN_20261024.md).

**2026-10-25 R1/R2 PE-Core T16 evidence reconciliation:** The independent
`codex/r1-pe-core-audit-20261021` branch's pinned checkpoint and CLEVR-4
artifacts were rechecked locally without touching that branch. A fresh isolated
CPU forward from the converted checkpoint returned finite image/text embeddings
and counted 69,546,113 parameters; this is runtime evidence, not task quality.
On the already observed synthetic CLEVR-4 reporting IDs, zero-shot similarity
scored 42.81% macro Accuracy and a frozen-feature candidate scorer reached
79.95%, with Count at 29.28%, below the historical Path A 88.06% macro result.
These are development-only results; no candidate promotion follows. A scoped
license reconciliation now marks the tested pinned OpenCLIP route ALLOW: its
checkpoint cards identify Apache-2.0 weights and OpenCLIP 3.3.0 is MIT. The
separate `perception_models` runtime package remains REVIEW/noncommercial and
was not used. PE-Core remains unsuitable for adoption on current evidence: the
tested checkpoint is 278 MB and task quality is weak. Revalidation details and
hashes are in
[`PE-Core T16 CPU revalidation`](docs/pretrained_reuse/R1_PE_CORE_T16_CPU_REVALIDATION_20261025.md);
the detailed task protocols/results remain linked from that report.

**2026-10-25 R2 CLEVR-4 spatial Count candidate audit:** An independent
VideoPrism spatial-token readout branch prepared a seed-17 split of 30,000
official-train images and 5,000 development images, excluding the prior
8,424/2,107 and 1,024/512 Path-A partitions. A new cross-experiment audit found
that **all 30,000 train and all 5,000 development IDs were included in an
earlier DINOv2 readout's 72,010-image supervised training set** (config train-ID
order SHA-256 `cc0c59c90ab363e46abfabd5910f7eab06d79c418da52650d741436f3df0d96c`).
Thus the files are identity-disjoint from the older Path-A split but are not
project-level unseen validation; do not use the 5,000 rows for independent
quality claims or candidate selection. The VideoPrism spatial-feature cache is
also incomplete: its validation array is absent and its training prefix stops
at 10,044 complete rows. No readout fit or quality metrics exist. The branch
report records Windows Application Control blocking its required VideoPrism
Python environment; do not bypass that control or use the partial cache. A new
independent task/split is required before another CLEVR-4 spatial-readout
selection. See
[`spatial Count readout report`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-spatial-count-readout-20261023/docs/pretrained_reuse/R2_CLEVR4_SPATIAL_COUNT_READOUT_20261023.md).

**2026-10-25 R2 Perception Test source-candidate diagnostic:** The independent
`codex/r2-perception-source-candidate` branch completed a seed-17 frozen-feature
readout for oracle event intervals and candidate object tracks: 56.93% top-1
set Accuracy / 1.2748 NLL on 606 development events from 80 videos. The selected
checkpoint was epoch 7; train loss continued falling while development NLL
rose after selection. Paired 10,000-resample video-group controls support
input dependence in this narrow task, including source-aligned versus offset
audio, but zero-video decisions are tie-sensitive. The partition comes from
the official training source, was inspected during preparation, has only 80
development videos, and does not establish participant independence or blind
generalization. This is exploratory oracle-assisted evidence; it does not close
the joint-modality gate or support product promotion. The saved best/final
states, training history and paired-bootstrap hashes were rechecked against
the report. See the [protocol and complete readout](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-perception-source-candidate/docs/pretrained_reuse/R2_PERCEPTION_AV_SOURCE_CANDIDATE_READOUT_20261025.md)
and [source preflight](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-perception-source-candidate/docs/pretrained_reuse/R2_PERCEPTION_SOURCE_CANDIDATE_PREFLIGHT_20261024.md).

**2026-10-25 R2 crop × coverage factorial analysis:** Added a CPU-only
hash-pinned analyzer for the four fixed-endpoint cells (center/full extent ×
repeat/fresh coverage), with per-modality Accuracy/NLL/Brier/ECE, macro metrics,
simple effects and an interaction contrast. It fails closed on missing cells,
changed validation IDs/order, prediction hash mismatch, or a non-endpoint run;
the incomplete preflight records no factorial metrics. Three existing 32,768-step
cells pass hash and shared-validation checks. The full-extent fresh fit is still
missing because the live GPU run owns the device; no new GPU job was started.
The full CPU suite passes (2,397 passed, 11 skipped); Ruff and manifest/catalog
validation pass. See the
[`crop × coverage factorial record`](docs/pretrained_reuse/R2_CROP_COVERAGE_FACTORIAL_20261025.md).

**2026-10-25 R2 full-extent × unique-coverage long comparison:** Added a
separate, hash-pinned fresh-sampling config and extended the full-extent runner
without changing its legacy default. The CPU-only pinned preflight regenerated
the same 32,768-step sample/LR schedule as the completed center-crop fresh arm,
with identical ordered validation and common 6,656-row parent probe. This adds
2,048 unique Text, Image and Audio questions but no Video questions. Targeted
tests (28) and Ruff pass. The GPU fit is pending because live PID26920 occupies
the RTX 3080 Laptop. An initial preflight writer attempt was rejected for a
receipt/file hash mismatch and preserved; fixed attempt-02 passes with all five
artifact hashes verified. No model-quality result is claimed. See
[`R2 full-extent unique-coverage long comparison`](docs/pretrained_reuse/R2_FULL_EXTENT_UNIQUE_COVERAGE_LONG_20261025.md).

**2026-10-25 R2 full-extent × unique-coverage result:** The guarded local
attempt-04 smoke, 32,768-update seed-17 fit, and best/final replay all passed;
replay max absolute difference was zero. The selected step was 4,096. On the
same development validation, step 512→32,768 raised macro Accuracy 29.63%→
47.23% but worsened macro NLL 1.730→3.413; Video Accuracy went 34.75%→33.00%
and Video NLL 1.354→9.102. The fixed hierarchical macro-NLL selector did not
improve after step 4,096. The complete 2×2 endpoint comparison found fresh
coverage helped several Image/Audio/Text scores but did not improve Video;
these are exposed, single-seed development results, not a blind evaluation or
product promotion. No sealed audit or cloud compute was used. Local tests and
Ruff passed, and GitHub Actions run 40198130727 is green for the recording
commit. See the completed
[`R2 long-fit report`](docs/pretrained_reuse/R2_FULL_EXTENT_UNIQUE_COVERAGE_LONG_20261025.md).

**2026-10-25 R2 long-fit runner repair:** Its first HotGPU smoke attempt stopped
at step 0 before model load because six files in the visual-cache provenance
receipt were compared against a newer checkout instead of the receipt's pinned
Git commit. All six expected hashes were independently verified from that exact
commit; no external worktree file was changed. The runner now records this
commit-blob verification and recognizes a capacity-checked HotGPU parent
reservation. Its only allowed auxiliary GPU PID is the verified Cursor GPU
renderer; unknown processes still block launch. A full-extent × fresh-coverage
fit has still not run; see
the report for the failure receipt and rerun conditions.

**2026-10-25 R2 UnAV-100 source/task screen:** UnAV-100 is a relevant
audio-visual event-localization task-design lead, but its official page says
raw video is retrieved from YouTube and that copyright remains with original
owners. No owner grant for model training was established, and the event labels
do not prove both audio and video are necessary. No data was accessed. Keep
REVIEW-only; it does not close the synchronized-AV gate. See the
[`UnAV-100 screen`](docs/pretrained_reuse/R2_UNAV100_SOURCE_TASK_SCREEN_20261025.md).

**2026-10-25 R2 SlideSpeech source/task screen:** SlideSpeech is a plausible
audio-plus-slide task-design lead for recognizing domain-specific spoken terms
with visual context. The official OpenSLR listing says CC BY-SA 4.0, but the
project site scopes that license to metadata; the official downloader retrieves
the underlying recordings from YouTube and no reviewed source grants rights to
those recordings for model training. The current release also lacks a proven
group-disjoint decision-task split. No data was accessed. Keep it REVIEW-only;
it does not close the both-input-required AV gate. See the
[`SlideSpeech source screen`](docs/pretrained_reuse/R2_SLIDESPEECH_SOURCE_SCREEN_20261025.md).

**2026-10-22 R2 AVE source/task screen:** The Audio-Visual Event dataset has
synchronized streams and temporal event boundaries, but its labels do not
establish that both modalities are necessary. The authors also do not own its
YouTube movie clips, and the official materials reviewed do not grant rights
for the underlying media or derivative training. No data was accessed. Keep
AVE review-only; it does not close the synchronized-AV task gate. See the
[`AVE source screen`](docs/pretrained_reuse/R2_AVE_SOURCE_SCREEN_20261022.md).

**2026-10-22 R2 FLARE joint-AV task/source screen:** FLARE's 53,580
hard-bimodal retrieval queries are a useful task-design lead: the benchmark
reports that vision-only and audio-only retrieval fail while joint AV retrieval
succeeds. It is not admitted for model training or evaluation. The release
contains media clips derived from Video-MME, whose official terms prohibit
commercial use and restrict copying/distribution; FLARE's dataset-level CC BY
declaration does not resolve underlying video rights. The viewer also presents
only a single `test` split for the inspected subset, without an established
group-disjoint train/validation/audit protocol. A few public preview caption
rows appeared during screening but were not retained or used. See the
[`FLARE source screen`](docs/pretrained_reuse/R2_FLARE_SOURCE_SCREEN_20261022.md).

**2026-10-22 R2 Ego4D Social public license-draft review:** The public draft
license now clarifies that qualifying derived models may be used commercially
after the applicable agreement is executed, while the dataset itself remains
restricted from redistribution or embedding in products. It is not an executed
license and does not authorize data access. The applicable licensor agreement,
TTM group counts, revisions, and media availability remain unresolved; no data
was accessed. Keep the source REVIEW. See the
[`license-draft review`](docs/pretrained_reuse/R2_EGO4D_SOCIAL_LICENSE_DRAFT_REVIEW_20261022.md).

**2026-10-22 R2 optional sensor typed JSON adapter:** The existing HARTH
sensor-window result now maps to the caller-ID Decision JSON contract while
preserving central-label timing and explicit abstention for incomplete input.
This is an API/runtime change only; no training or new quality evaluation was
performed, and the optional sensor route does not close any required modality
or joint-fusion gate. See
[`sensor typed JSON adapter`](docs/pretrained_reuse/R2_NATIVE_SENSOR_TYPED_JSON_20261022.md).

**2026-10-22 R2 Path A typed JSON runtime adapter:** The Path A dispatcher can
wrap existing text, image, audio, and video option probabilities in the stable
caller-ID Decision JSON contract without changing scoring or adding model
parameters. Targeted CPU tests passed (51); the later compatibility fix kept
raw-dispatch parity probes working, and the full suite passed (2,380 passed,
11 skipped). A one-observation-per-modality CPU runtime replay found typed JSON
consistent with runtime outputs for all four paths. Audio's saved-reference
probability delta exceeded the CUDA-oriented `1e-5` parity limit on CPU, despite
matching predictions; this is runtime numeric parity, not a quality finding.
This does not establish calibration or cross-modal reasoning. Full detail:
[`Path A typed JSON replay`](docs/pretrained_reuse/PATH_A_TYPED_JSON_DISPATCHER_20261022.md).

**2026-10-22 R2 AVA Active Speaker label/split follow-up:** Explicit train and
validation archives contain 120 and 33 video groups, with 2,676,314 and
768,307 annotation rows, respectively; train/validation video overlap is zero.
The 33-group validation is below the proposed 200-group quality gate, and
underlying YouTube media rights remain unresolved, so no fit or quality
evaluation is admitted. An inventory wildcard inadvertently enumerated metadata
for an additional test-named archive; no member payloads or labels were read or
used, and that archive is excluded from this work and cannot serve as a blind
audit here. See the [label and split audit](docs/pretrained_reuse/R2_AVA_ACTIVE_SPEAKER_LABELS_SPLIT_AUDIT_20261022.md).

**2026-10-22 R2 synchronized-AV source screen:** Ego4D Social `Talking to me`
is now recorded as a review-only candidate for a supplied-face-track decision
combining audio and video. Its official task uses active-speaker/face tracks
and labels whether speech is directed to the camera wearer, but data and
annotation access requires acceptance of an agreement whose terms were not
available in this screen. We did not access annotations or media. Exact usable
groups, rights and local subset availability remain unknown. Pinned code review
found the TTM branch expects AVD train/validation annotations, while its README
still says benchmark data details are forthcoming; old AVD v1 counts are not
verified TTM counts. No training or evaluation is admitted. See the
[source screen](docs/pretrained_reuse/R2_EGO4D_SOCIAL_SOURCE_SCREEN_20261022.md)
and AV7 entry in
[`R0_TASK_BATTERY_DRAFT_20261021.md`](docs/pretrained_reuse/R0_TASK_BATTERY_DRAFT_20261021.md).

**2026-10-22 R2 AV temporal readout scale follow-up:** Compared audio-only,
pooled AV, and eight-frame LSTM AV readouts on nested 96/224-group training
cohorts and the same 64 fresh video groups. All six fixed 2,048-update runs
completed on CPU; all selected initialization at step 0 after the heads
memorized training data by step 32 and validation NLL degraded. Increasing to
224 groups did not prevent the overfit, and sequence AV did not establish a
gain over pooling. The paired 5,000-resample intervals include zero. Treat this
as a narrow development-task result, not a model-family ceiling or product
quality result. No official test/sealed audit was opened. See the
[frozen protocol and results](docs/pretrained_reuse/R2_AV_TEMPORAL_SCALE_V2_RESULTS_20261022.md)
and [learning curves](docs/pretrained_reuse/R2_AV_TEMPORAL_SCALE_V2_CURVES_20261022.csv).

**2026-10-22 R2 HoloAssist fixed-checkpoint replay:** Replayed the already
selected 1,042-session-train joint, audio-only, and video-only readouts on the
same 207 official-validation sessions used in the earlier pilot. This was
external to the candidate's train and selection groups (zero session/media
overlap), but is not blind because the earlier pilot selected checkpoints on
these sessions. Joint AV reached 77.11% Accuracy / 0.5034 NLL versus audio-only
76.00% / 0.5061; paired session intervals include zero. Joint is clearly better
than video-only, but the result does not show that video is required or reach
the quality goal. No fit or tuning occurred, and official test/sealed audit
remained closed. All official train and validation groups in this line are now
exposed, so new HoloAssist experiments require a different admitted source.
See the [fixed-checkpoint replay report](docs/pretrained_reuse/R2_HOLOASSIST_OFFICIAL_VALIDATION_FROZEN_REPLAY_20261022.md).

**2026-10-21 R1/R2 VideoPrism temporal-token runtime probe:** The pinned
VideoPrism-B CPU forward on one fixed 8-frame development-train clip returned
finite `[1, 2048, 768]` tokens, i.e. 256 spatial tokens per frame after its
temporal encoder. The existing AV extractor mean-pools all 2,048 tokens to
`[768]`, so the earlier small-readout results did not test an explicit
sequence-preserving readout. This proves the representation can be exposed,
not that a recurrent/gated head will improve task quality. No predictions,
metrics, fitting, or validation selection were run. Full audit and retained
attempts: [VideoPrism temporal-token probe](docs/pretrained_reuse/R1_VIDEOPRISM_TEMPORAL_TOKEN_SHAPE_20261021.md).

The extractor now has opt-in `per_frame_mean` (`[T,768]`) and
`spatiotemporal_tokens` (`[T,N,768]`) feature layouts; legacy global pooling
remains the default. A one-row, label-free CPU model smoke generated `[8,768]`
and validated terminal run metadata. Unit tests cover ordering, shapes, finite
values, and compatibility with the old pooling path. A fresh official-TRAIN
same-video AV temporal matching holdout was then extracted and fit with matched
audio-only, pooled-AV, and LSTM sequence heads. All three heads memorized 96
training groups by step 32; validation NLL worsened and the fixed selector
retained initialization (step 0). The sequence head did not establish a
temporal-order or generalization gain over pooling. This small, validation-
selected development study does not establish a representation ceiling or
product quality. Preserve its weak/inconclusive result and do not tune on its
32 validation groups. [Protocol, curves, metrics, controls and hashes](docs/pretrained_reuse/R2_AV_TEMPORAL_MATCH_READOUT_20261021.md).

**2026-10-21 22:26 JST R0 live recheck:** Refreshed origin and checked all
40 registered worktrees plus open PRs #9–#20 without editing other checkouts.
Six worktrees remain dirty, including the user's primary Teacher checkout; all
were preserved. PR #15 remains Draft/Open at `e38447c`; PR #16–#20 remain open
on their independent branches. Orbitune's supervised training child PID 28928
is still active on the RTX 3080 Laptop (15,821/16,384 MiB used, 39% utilization);
no Tiny Omni GPU job was launched. C: has 12,294,930,432 bytes free. The latest
R2 Perception Test audio-video transfer result remains weak/inconclusive for
joint Accuracy (+0.78 points over Video; paired interval -4.30 to +5.86), so
the M-2 both-required task gate remains open. Full snapshot and exact dirty
worktree paths: [R0 live audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261021_2226JST.md).

**2026-10-21 R2 typed Decision JSON foundation:** Ported the stable, caller-ID
based `choice` / `yes_no` / `ordinal_score` serializer and schema note from the
independent PR #20 into this roadmap branch. The legacy serializer is retained;
the interface adds no model parameters and does not claim calibration or task
quality. Targeted tests (17) and Ruff pass. PR #20 remains open and untouched.

**2026-10-21 R2 Perception Test fixed-readout train-transfer diagnostic:** The
frozen Audio, Video, and Joint readouts were scored once on 256 new,
video-disjoint official-train groups (128 per visibility label). Accuracy was
53.52% Audio, 65.23% Video, and 66.02% Joint. Joint minus Video was +0.78
points (paired 95% bootstrap interval -4.30 to +5.86); this does not establish
an Accuracy gain over Video alone. Joint NLL/Brier/ECE were better on this
selected cohort, but calibration remained poor. This is nonblind, same-source
train transfer, not independent validation or product evidence. Attempt 01's
250ms margin failed to guarantee the fixed 2s window; the corrected attempt 02
requires 1s context and retains the first attempt. Full protocol, metrics, and
hashes: [transfer report](docs/pretrained_reuse/R2_PERCEPTION_VISIBILITY_TRAIN_TRANSFER_20261021.md).

**2026-10-21 R2 AVQA rights recheck:** Tsinghua's official AVQA page describes
four-choice `View`/`Sound`/`Both` questions across 57,335 manually annotated
items, making it a strong task-design lead. At pinned official repository
revision `a5056c1`, GitHub reports no repository license and the root tree has
no dataset license file; underlying clips come from VGGSound/YouTube. Keep it
`REVIEW`; no annotation contents or media were fetched, and no fit is admitted.
[Rights recheck](docs/pretrained_reuse/R2_AVQA_RIGHTS_RECHECK_20261021.md).

**2026-10-21 21:18 JST R0 live recheck:** PR #15 is still Draft/Open at
`e92dbbd` with both CI checks green. Forty worktrees are registered and six
remain dirty; the primary Teacher worktree was left untouched. Orbitune's
cooperatively supervised BF16 training process still owns 15,821 MiB of the
RTX 3080 Laptop's 16,384 MiB, so no Tiny Omni GPU job was started. The current
source screens still have not admitted a sufficiently grouped, rights-cleared
task that proves both synchronized inputs are necessary. Continue CPU-only
work until this data/task gate and GPU availability permit the next preregistered
fit. [Snapshot](docs/pretrained_reuse/R0_LIVE_AUDIT_20261021_2118JST.md).

**2026-10-21 R2 CaptainCook4D media/group recheck:** The text+video step
consistency task remains provisional. Five sampled training GoPro links and two
sampled synchronized HoloLens PV links returned 404. One previously acquired
training video still passes a hash-checked CPU frame-decoding smoke, but official
training has only six participant groups (203 recordings after removing the
known train/validation identity overlap), and validation has two participants.
No quality fit or test evaluation is justified from the current accessible data.
Do not use test records to fill the group gap; seek a working official media
source and a new participant-held-out development design, otherwise continue
the R2 source search. [Recheck](docs/pretrained_reuse/R2_CAPTAINCOOK4D_MEDIA_GROUP_RECHECK_20261021.md).

**2026-10-21 R2 Perception Test temporal-match experiment:** Built a
video-grouped, development-only two-choice task from the pinned Perception Test
train/validation annotations: choose the same-video/same-label audio window that
matches an 8-frame video interval. The frozen pooled joint readout scored 47.60%
on 313 previously inspected validation videos, below audio-only at 50.80%; the
paired 2,000-resample video-group interval for the difference included zero.
Within-label feature shuffles and mean-imputed video did not hurt the pooled
joint readout. One 19,009-parameter conditional gate completed 2,048 updates but
overfit by step 32; its fixed selector chose initialization at step 0. This
does not test a temporal gate: the readout received pooled feature vectors and
discarded temporal tokens. The task does not demonstrate complementary AV
quality and leaves R2 M-2 open. The validation is not blind, and no sealed audit
or official test split was used. [Protocol and full results](docs/pretrained_reuse/R2_PERCEPTION_AV_TEMPORAL_MATCH_20261021.md).

**2026-10-21 R0 task-battery evidence reconciliation:** The task-battery
proposal now records the one-time Speech Commands official-test comparison
(97.72% baseline / 97.86% train-only temperature candidate; no clear Accuracy
difference), the fresh CLEVRER reasoning holdout result (both candidates near
binary chance with paired intervals including zero), and the matched
32,768-update coverage comparison (selected fresh arm slightly worse on macro
metrics; late endpoint overfit; no new Video questions). These splits are
observed and closed to further model selection. The narrow audio result does
not establish general speech, and the video results do not establish a model
family ceiling. See the additive [task-battery update](docs/pretrained_reuse/R0_TASK_BATTERY_DRAFT_20261021.md).

**2026-10-21 R2 multimodal source admission screen:** Screened official ZOD
and IIT Audio-Visually Indicated Actions materials without downloading media.
ZOD has camera/LiDAR/radar/IMU but requires an emailed access request and its
sequence annotations are central-frame only; no both-required task or encoder
pretraining overlap is established. IIT's 378 synchronized AV sequences cover
9 participants and access is limited to institution-linked noncommercial
research requests; its action labels are intentionally audio-distinctive.
Neither is admitted for a local quality run. See the [source screen](docs/pretrained_reuse/R2_MULTIMODAL_SOURCE_ADMISSION_SCREEN_20261021.md).

**2026-10-28 ZOD follow-up:** Open Draft PR #35 verifies shared annotation UUIDs
link camera 2D and LiDAR 3D boxes, making a narrow supplied-candidate
association task schema-feasible. Access-request versus public-torrent
authorization, CC BY-SA model implications, drive-disjoint group counts,
pretraining overlap, and both-input necessity remain unresolved. ZOD remains
unadmitted; this supersedes only the earlier statement that box co-reference
was not established. See the [cross-branch reconciliation](docs/pretrained_reuse/R2_ZOD_CROSS_BRANCH_RECONCILIATION_20261028.md).

**2026-10-21 R2 synchronized sensor/video source screen:** Compared CUHK-X,
C-MHAD, and XRFv2 Plus from official publisher materials. CUHK-X is the most
promising bounded research candidate (seven aligned modalities, 64,267 samples,
30 participants), but it requires a noncommercial, non-redistributable DUA and
does not yet establish a both-input task; 30 participants also falls below the
provisional 200-group floor. C-MHAD has 12 participants and no separate data
license on its publisher page. XRFv2 Plus has 16 volunteers and an unspecified
dataset license. No data was accessed and none is admitted to fitting. See the
[source screen](docs/pretrained_reuse/R2_SENSOR_VIDEO_SOURCE_SCREEN_20261021.md).

**2026-10-21 R2 optional sensor source gate:** UCI PAMAP2 metadata was screened without downloading the archive. UCI states CC BY 4.0, but the dataset has only 9 subjects and the audited MOMENT materials do not provide a complete pretraining item manifest that proves PAMAP2 was excluded. Overlap is UNKNOWN; no MOMENT score on PAMAP2 may select a candidate. Keep sensor work optional and research-only until a sufficiently grouped, provenance-auditable task is found. See the [PAMAP2 admission screen](docs/pretrained_reuse/R2_PAMAP2_SENSOR_ADMISSION_SCREEN_20261021.md).

**2026-10-21 R2 synthetic AV timing pilot reconciled:** A seed-17, four-arm
frozen-feature run completed 2,048 updates per arm on 1,596 paired train rows
and 390 development rows grouped by CLEVRER scene. On the generated-tone task,
joint gated LSTM scored 78.97% Accuracy / 0.4566 NLL versus audio-only 72.31% /
0.5051; text-only, video-only and temporal-mean controls were at chance. The paired
79-scene bootstrap supports a development-set gain, but the same validation
selected checkpoints, the synthetic audio carries only binned energy, and final
joint training overfit (98.06% train Accuracy; final validation NLL 0.9432).
Treat this only as controlled synthetic temporal-binding evidence. The exposure scan found 405 untouched candidate scenes and the pilot consumed them; any expansion must reuse previously-seen official-train scenes or screen another source. [Protocol and full metrics](docs/pretrained_reuse/R2_CLEVRER_SYNTHETIC_AV_RESULTS_20261020.md).

**2026-10-21 R2 AVID-Bench source screen:** AVID offers a relevant audio-visual
inconsistency task, but the exact source-media rights and a pinned TRAIN-only
release are not verified; accessible Hugging Face metadata currently lists test
annotations only. The README estimates 84 GB for test media and 415 GB for the
full dataset, so no media or test annotations were downloaded or opened. Keep
the source `REVIEW`; do not treat its model results as this project's evidence.
Reopen only with exact media rights, a hash-pinned TRAIN release, and a small
local subset. [Source screen and gate](docs/pretrained_reuse/R2_AVID_SOURCE_SCREEN_20261021.md).

**2026-10-21 14:43 JST R0 live GPU / HotUGPU recheck:** Orbitune
training PID 10912 (parent 24896; supervisors 4696 and 23856) is still
running. RTX 3080 Laptop reports 15,821/16,384 MiB used, 355 MiB free, and
100% utilization. No Tiny Omni GPU job was launched, and no Orbitune process,
checkpoint, or data was touched. The existing HotUGPU guard waits for the GPU
to yield, reserves VRAM, then starts its child under a minimum-free-memory
gate; this can block Orbitune's configured 14,000 MiB free-memory watchdog
from launching a new child while the reservation is held, but does not stop a
currently running Orbitune child. Continue CPU-only work until the GPU yields;
any later CUDA job must use the guard and a separate artifact directory.

**2026-10-21 R2 AMI selected-video full-decode follow-up:** On the
AMI source-screen branch, CPU-decoded all 20 AVI streams from the first five
selected SA/SB meetings that previously had only sampled-frame checks. File
size/SHA-256 matched the pinned preflight for all 20; there were no internal
PTS gaps, and all 70 current candidate windows end before their mapped camera's
last decoded frame. Three files have 18/49/47 fewer terminal frames than their
container declarations; these are recorded, not silently normalized. Together
with the prior 28 streams, this accounts for all 48 selected AVI streams, but
does not prove AV synchronization or visible-speaker labels. AMI remains
review-only; no GPU/model/SC access. Report and artifact hashes are in
[AMI screen commit `bbd1cb9`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-ami-av-source-screen-20261021/docs/pretrained_reuse/R2_AMI_CROSSMODAL_SPEAKER_MATCH_SCREEN_20261021.md).

**2026-10-21 R2 AMI development visibility gate:** A development-only,
three-frame-per-window review covers all 111 preliminary SB candidates from
ES2003a, ES2011a, IS1008a, TS3004a and TS3006a. Results: 69 clear faces, 6
limited views, 2 visible people without a clear face, and 34 samples with no
clear person in the target camera. TS3004a and TS3006a account for 14/30 and
18/27 no-person samples. This weakens the current weak-label construction;
AMI stays review-only pending stronger visible-speaker and synchronization
evidence. No SC material or audio was opened and no GPU run was started. The
CPU manifest builder and tests are in commit `406f003` on the AMI source-screen
branch; the [detailed screen report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-ami-av-source-screen-20261021/docs/pretrained_reuse/R2_AMI_CROSSMODAL_SPEAKER_MATCH_SCREEN_20261021.md)
contains hashes and limits.

**2026-10-21 R2 native sensor results reconciliation:** The existing seed-17
HARTH subject-disjoint sensor-only fits were located and their selected saved
predictions independently recomputed; no duplicate run was started. Mean-pooled
temporal Conv1d achieved 91.47% Accuracy / 73.26% balanced Accuracy, while a
smaller static-statistics control achieved 93.87% / 69.80% on the same 12,763
windows from only three validation subjects. The temporal model dropped to
84.64% Accuracy after within-window time shuffle; the static control was
unchanged. Class-2 recall and mixed-transition performance remain poor, and
validation selected the checkpoints. This is a narrow engineering result, not
independent sensor quality; no calibration/sealed cohort was accessed. Full
metrics, learning curves, hashes and limitations are in
[`R2_NATIVE_SENSOR_SUBJECT_RESULTS_20261012.md`](docs/pretrained_reuse/R2_NATIVE_SENSOR_SUBJECT_RESULTS_20261012.md).

**2026-10-21 R1 PR review follow-up:** The unresolved PR #16 review finding
on PE-Core T16 release access was corrected in the dedicated PR branch at
`afea6624dfe247350ecbb704d61099304009f2a0`. The pinned checkpoint and hash
are verified, so its inventory release field now says `pretrained and
accessible`; license status remains `REVIEW` and runtime support remains
conservative. Targeted pytest passed (7 tests) and Ruff passed. GitHub returned
no status checks for the commit; the inline review thread is outdated but still
unresolved. No other PR/worktree was changed.

**2026-10-21 13:37 JST R0 live-state recheck:** The roadmap/continuation
worktree is at `dc60ad9` before this audit; PR #15 remains draft/open at
`71c4db1`. PRs #16–#19 are also open for the PE-Core audit, MAVD source screen,
CaptainCook4D screen and AMI source screen. Existing Teacher/QAT PRs #9–#14
remain open and untouched. The separate user checkout is on
`codex/teacher-quality-clean-dev-v1` with pre-existing modified/untracked files;
this audit did not edit them. Orbitune's supervisor PIDs 4696/23856 and its
training PIDs 22580/17812 were live. The RTX 3080 reported 15,821/16,384 MiB
used, 355 MiB free and 77% utilization; C: had 20.60 GiB free. No process was
signalled, no GPU experiment was launched, and no live checkpoint was read or
hashed. R0 still lacks a selected end-user device and frozen release battery;
R2 M-2 still lacks an admitted both-required task. Continue source/task admission
and CPU-only validation until those gates and GPU availability permit a
preregistered run. Full process commands, PR SHAs and snapshot evidence are in
[`R0_LIVE_AUDIT_20261021_1337JST.md`](docs/pretrained_reuse/R0_LIVE_AUDIT_20261021_1337JST.md).

**2026-10-21 R7 Path A image runtime full-validation parity:** Replayed all
2,048 saved decisions across the 512-image checkpoint-selection validation
slice using the manifest-only image runtime bundle on JAX CPU. There were zero
prediction mismatches and maximum absolute probability delta was 4.98e-6
(within 1e-5); inference took 547.27 seconds and peak working set was about
4.26 GB. This confirms packaging/runtime parity only on previously inspected
validation data, not independent image quality. See the
[replay record](docs/pretrained_reuse/R7_PATH_A_IMAGE_FULL_VALIDATION_RUNTIME_PARITY_20261021.md).

**2026-10-21 R2 text-free audio calibration check:** A speaker-disjoint
temperature fit on 264 train-only examples left Accuracy unchanged at 96.56%
on the already-inspected 640-example Speech Commands development split, while
NLL/Brier/ECE moved from 0.3346/0.1170/0.1885 to 0.1435/0.0559/0.0245.
This is a focused calibration result, not blind evidence or general speech
understanding. The calibration-trained readout saw 1,016 rather than the prior
model's 1,280 training examples. A retrospective 10,000-resample, 206-speaker
bootstrap on that previously inspected validation set has NLL/Brier/ECE
improvement intervals excluding zero; Accuracy is unchanged by construction.
See the [protocol and result](docs/pretrained_reuse/R2_TEXT_FREE_AUDIO_TEMPERATURE_20261021.md).

**2026-10-21 R0 task-battery draft:** Existing text/image/audio/video and
synchronized AV evidence is mapped to candidate task slices, group rules and
the per-task >=95% target. The Windows/RTX 3080 remains an engineering host;
end-user device and release calibration/latency thresholds remain undecided.
This draft does not freeze a release battery or authorize a new fit. See the
[task-battery proposal](docs/pretrained_reuse/R0_TASK_BATTERY_DRAFT_20261021.md).

**2026-10-21 R2 Path A text readout CPU replay:** Re-loaded the pinned
196,865-parameter Decision readout and rescored all 2,540 existing cached
validation feature rows on CPU. Accuracy/NLL/Brier/ECE-15 matched the selected
run within 1.1e-7 absolute; this verifies head serialization and cached-feature
scoring only. The frozen encoder was not run, the validation is previously used,
and 49.37% Accuracy remains weak synthetic-only quality. See the
[CPU replay record](docs/pretrained_reuse/R2_PATH_A_TEXT_READOUT_CPU_REPLAY_20261021.md).

**2026-10-21 R7 Path A package integrity recheck:** All 42 manifest-listed
files (687.25 MB, 170.50M unique parameters) revalidated by size and SHA-256.
Strict package verification finds two unlisted Python bytecode cache files
under the image runtime; their origin is unknown and existing package contents
were preserved. A separate manifest-only copy has the same manifest hash and
passes strict verification. The provisional 200 MB target remains unmet. A
hypothetical storage-only INT8 estimate is not a quantized runtime or a quality
result. See the [package integrity record](docs/pretrained_reuse/R7_PATH_A_PACKAGE_INTEGRITY_RECHECK_20261021.md).

**2026-10-21 R0 live-state refresh:** Current `main` remains `09246d7`; PR #15
remains Draft/Open and the research branch is separate. Thirty-four worktrees
were inventoried; six are dirty, including the user's primary Teacher worktree,
which was left untouched. RTX 3080 Laptop is currently occupied by Orbitune
(15,373/16,384 MiB); no Tiny Omni GPU work was started. The current audit also
reconciles the existing 388-video Perception source-correspondence results and
PCH recording-LSTM result; do not duplicate them. Path A v4 remains 170.50M
parameters / 687.25MB, so the provisional 200MB package target is unverified.
Task battery, per-task quality/calibration gates, deployment device and local
runtime limits remain open. See [the additive R0 snapshot](docs/pretrained_reuse/R0_LIVE_AUDIT_20261021.md)
and [machine-readable inventory](docs/pretrained_reuse/R0_LIVE_AUDIT_20261021.json).


**2026-10-21 R2 Perception Test source and matched-input screen:** A fixed
seed-17, 128/class, video-disjoint cohort of 256 train / 256 validation AV events
passed acquisition, synchronization, and cross-split media/feature hash gates.
EfficientAT audio and VideoPrism video frozen features were evaluated with one
fixed small linear readout each and a joint readout. Balanced validation Accuracy
was 56.25% audio, 57.81% video, and 64.84% joint; all three fit 100% train
Accuracy, signaling severe overfit. Joint predictions react to both streams, but
this single narrow visibility task does not establish general multimodal
complementarity; NLL/Brier evidence versus audio is uncertain. This candidate
falls far short of the 95–98% aspiration and is not a product candidate. No
training of encoders, quantization, test evaluation, or sealed-audit access.
Keep the readout closed; pursue independent task/data diagnosis before larger
model work. Full protocol, uncertainty, hashes, and logging caveat in
[the source screen report](docs/pretrained_reuse/R2_PERCEPTION_TEST_SOURCE_SCREEN_20261021.md).

**2026-10-21 R2 AV label-support audit:** A hash-pinned CPU-only join of saved
predictions and source annotations found strong event-category/visibility
associations in the balanced 256-video development cohort. A corrected
distinct-video count in the full source annotation sets finds 229 train and 190
official-validation videos containing both visibility states within an event
category; these are not adjudicated counterfactual pairs. This strengthens the warning that the current
readout cannot establish both-input necessity; no fitting, selection, media
decode or test access occurred. Keep the task exploratory until a fresh,
predeclared video-grouped conditional design is available. See the
[additive label-support report](docs/pretrained_reuse/R2_PERCEPTION_AV_LABEL_STRATA_20261021.md).

**2026-10-21 R2 Perception AV scale and input-use follow-up:** The matched
expanded-fit comparison added 233 fit videos / 550 events while retaining the
same 2,048-update budget. Joint development Accuracy changed 33.56%→32.88%
(-0.7 pp; paired video-bootstrap interval spans zero); NLL improved by 0.041,
while Brier/ECE worsened. Parameter-matched video-only Accuracy rose 4.8 pp,
but its interval also spans zero and calibration worsened. A separate
fixed-checkpoint intervention found worse NLL when audio came from another
video or the previous event; zero-audio change was uncertain. This establishes
some alignment sensitivity, not correct audio-semantic use or general fusion
benefit. Both diagnostics use development cohorts and are not blind. Reports:
[coverage scale](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-perception-av-scale-v1/docs/pretrained_reuse/R2_PERCEPTION_AV_EVENT_SCALE_RESULTS_20261021.md)
and [fixed-checkpoint input controls](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-perception-av-scale-v1/docs/pretrained_reuse/R2_PERCEPTION_AV_EVENT_INPUT_CONTROLS_SCALE_20261021.md).

**2026-10-21 R2 audiovisual source screens:** BatVision remains REVIEW: its
five CC-BY-SA 4.0 shards total 168.5 GB; group integrity and a both-required
task remain unproven. TAU Urban Audio-Visual Scenes 2021 also remains REVIEW:
the official Zenodo metadata says `other-nc`, and its development archive is
107.7 GB. Neither source was downloaded. See the
[BatVision screen](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-batvision-source-screen/docs/pretrained_reuse/R2_BATVISION_SOURCE_SCREEN_20261021.md)
and [TAU screen](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-batvision-source-screen/docs/pretrained_reuse/R2_TAU_AVSC_SOURCE_SCREEN_20261021.md).
R2 M-2 remains open; synchronized media or prediction sensitivity alone is not
enough to admit a joint-quality run. Continue with rights-cleared, independently
grouped data and a task-level test that requires multiple inputs.

**2026-10-25 R2 synthetic AV input controls:** A fixed-checkpoint CPU replay
and five input perturbations on the already exposed CLEVRER synthetic AV v2
development validation showed sensitivity to video presence, scene identity,
temporal order, and audio timing. The 240-scene paired bootstrap intervals for
Accuracy and NLL differences excluded zero for all interventions. This remains
post-selection synthetic sensitivity evidence, not blind or natural-AV
validation; R2 M-2 stays open pending a rights-cleared task that establishes
multi-input necessity. See
[`R2_CLEVRER_SYNTHETIC_AV_CONTROLS_20261025.md`](docs/pretrained_reuse/R2_CLEVRER_SYNTHETIC_AV_CONTROLS_20261025.md).

**2026-10-18 R0 provisional engineering contract:** The RTX 3080 Laptop
workstation is now recorded as an engineering reference host, not the selected
deployment device. Preserve the full pre-quantization <=200M parameter target
and separately account for the provisional <=200MB complete quantized package;
quantization does not reduce the parameter count. For each predeclared required
task, >=95% Accuracy remains the product goal and 98% the aspiration, but this
is not a release gate until the task battery, independent-test uncertainty,
calibration/abstention policy and target device are frozen. Compute remains
local with $0 cloud spend; each run must preregister measured time/storage
budgets and report device use. No universal latency/energy limit is invented.
Joint-task admission requires truth-bearing evidence of complementary input
necessity; replacement-induced output changes establish sensitivity only. R0
remains partly open. See the [provisional contract](docs/pretrained_reuse/R0_DECISION_MATRIX_20261011.md).

**2026-10-18 conditional AV gate fit reviewed:** A separate branch completed a
matched seed-17, 32,768-update E0G1 run. The gate outputs collapsed below 0.01
for all valid options by update 512 and stayed near zero; the added gate has
61,634 parameters (about 0.031% of the pre-quantization system), so compute
capacity is not the issue. E0G1 gained 5.07 pp joint-view Accuracy over E0G0,
but the paired NLL interval includes zero and its video-only Accuracy fell
3.83 pp. The correction path still receives the query, so these results do not
show active gate-based AV evidence use. A fixed-checkpoint replacement
diagnostic is now complete: donor-audio/video substitutions changed the
argmax on 38.01%/56.19% of 8,875 validation questions. The query-only and full-
residual sensitivities match at aggregate precision; this establishes output
sensitivity, not correctness or causal AV use. Do not train another gate
variant from this result alone. See the [matched gated fit and uncertainty
report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_GATED_POSTNORM_LONG_FIT_20261018.md)
and [replacement diagnostic with hashes and per-task results](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md).

**2026-10-18 AV replacement diagnostic preflight and completion:** Before retrying the
inference-only replacement diagnostic, the separate `codex/r2-av-postnorm-stability`
branch is now at `216b16bde7299c60915052159631f1cc3868e602`. The runner polls
CUDA process IDs at most every two seconds instead of launching `nvidia-smi`
before each of 44,375 forwards. It records baseline, query-only and full-
residual logits/sensitivity in each diagnostic pass without changing the
training-pinned model implementation; all 1,115 run-time training-source hashes
matched. Attempt 08's 109/44,375 partial output remains preserved and is not a
metric. The full CPU suite passes 2,389 tests (11 skipped), Ruff passes, and
GitHub Actions passed for code commit `db39767bf3a9008fa31e1387a404173c059e4dd1`.
The runner completed attempt 11 after cooperative GPU yield, checking all
44,375 forwards, 17,750 exact self replacements, 2,755 input pins and 1,120
source pins. Peak VRAM was 116.8 MiB; no optimizer update or sealed-audit
access occurred. Attempts 08–10 remain preserved. See the [runner preflight and limits](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_REPLACEMENT_RUNNER_PREFLIGHT_20261018.md)
and [completed result](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md).

**2026-10-18 R0 live-state reconciliation:** PR #15 remains draft/open; the
roadmap worktree is clean and synchronized with its remote. The only manifest
hash change from the original R0 snapshot is the intended LibriSpeech
hard-negative evaluation-catalog entry; Open-Jev test metadata is unchanged.
At the 06:44 JST audit snapshot, the RTX 3080 was idle while Orbitune's
supervisor remained alive. Later, the fixed AV inference-only diagnostic used
the authorized cooperative yield path; no Orbitune process or file was
interrupted or modified. The current AV corpus still has no admitted
both-required paired task, and R0 device/task acceptance criteria remain open.
See the [dated audit](docs/pretrained_reuse/R0_LIVE_AUDIT_20261018.md).

**2026-10-18 CLEVRER unique-scene coverage comparison completed:** At a fixed
2,048 updates, expanding the frozen VideoPrism + MiniLM readout's train pool
from 256 to 768 unique scenes improved selected macro Accuracy 38.9%→45.9%
and NLL 1.183→1.132 on the same 256-scene development validation set. Paired
scene-bootstrap intervals support the pooled change, driven by static
descriptive questions; temporal metrics remain uncertain. Exact question
templates recur across scenes, so this is not template-independent or blind
validation. No promotion; [full results and retained failures](docs/pretrained_reuse/CLEVRER_UNIQUE_COVERAGE_V2_RESULTS.md).

**2026-10-18 text-free CLEVRER temporal probe completed:** One fixed-seed
mean-MLP/BiLSTM comparison completed 2,048 updates per arm. Selected temporal
NLL/Brier modestly favor BiLSTM; its Accuracy difference is uncertain, static
metrics do not establish a gain, and both arms show severe late overfitting.
This is a narrow typed-program video probe, not multimodal product evidence.
Keep the recurrent block unadopted pending replication and a representative
task mix. See the [full report](docs/pretrained_reuse/R2_TEXTFREE_VIDEO_TEMPORAL_RESULTS_20261018.md).

**2026-10-17 text-free audio Decision probe completed:** A CPU-trained linear
head over the existing frozen Whisper feature cache classifies the fixed
10-keyword Speech Commands taxonomy without question/candidate text or a text
encoder. On the previously observed 640-row validation split it scores 96.56%
Accuracy / 0.2834 NLL / 0.0986 Brier / 0.1515 ECE; the previous Audio+Text
readout scored 95.31% / 0.1624 / 0.0661 / 0.0377 on those same IDs. Thus the
text-free head is slightly more accurate but substantially less calibrated in
this narrow development task. It uses 3,850 trainable readout parameters and
passes exact save/reload; no blind, open-vocabulary or product claim follows.
This partially addresses R2 M-0/M-1 audio only; image/video/screen/sensor and the
required metadata task battery remain open. Clarification: 61,634 parameters are
the conditional gate MLP itself; 843,075 counts the full new AV residual path.
No AV joint task has yet justified training that path. See
[text-free audio Decision report](docs/pretrained_reuse/R2_TEXTFREE_AUDIO_DECISION_20261017.md).

**2026-10-17 frozen AV cache and long-fit evidence verified:** The full video bank
finished all 11,151 events / 409 parents / 32,582 questions; a read-only rehash
matched all 1,227 stored files (70,430,943,552 bytes) and the 21-event reference
replay. The audio/query producer's terminal receipt is `INCOMPLETE` because its
expected smoke-reference count was wrong; a separate full revalidation passed
all inputs and 22,302 stored audio-feature digests, while explicitly not claiming
a full fresh-encoder replay. The 21-event audio smoke replay and real feature
transport replay passed. A separate 32,768-update post-attention-normalization
candidate improved the fixed video validation view from 38.45% at step 512 to
61.84% at final (selected step 16,384); its final four-modality macro NLL was
worse than initialization, and this is not evidence that a learned AV gate helps.
No sealed audit, quantization or product promotion occurred. Preserve the cache
and run; next admit genuinely both-required, independently grouped AV tasks and
define task/device release thresholds before another quality-selected run.
[Cache completion, integrity, run references and open gates](docs/pretrained_reuse/R2_GREATEST_HITS_CACHE_COMPLETION_20261017.md).

The original 21-event smoke and full-cache protocol/results remain preserved:
[full-extent protocol](docs/pretrained_reuse/R2_GREATEST_HITS_FULL_EXTENT_PROTOCOL_20261017.md) ·
[historical launch and early progress record](docs/pretrained_reuse/R2_GREATEST_HITS_FULL_EXTENT_RESULTS_20261017.md).

**2026-10-17 fixed TRAIN source review completed:** Decode and inspect one fixed
event from each of 21 TRAIN dates (315 frames, 84 payloads). Preserve all labels,
including ambiguous material/reaction and edge-contact cases. Admit only scoped
native material/action/reaction development; silent-video-derived labels do not
certify both-required truth, precise physical sync or unseen-actor/site quality.
Next execute full-extent encoder/replay on these same events before common caching
and matched Attention/exchange/gate learning. All compared arms must share the new
geometry; valid counterfactual labels and product admission gates remain open.
No fit, quality claim, audit access or product promotion in this review.
[Evidence and task disposition](docs/pretrained_reuse/R2_GREATEST_HITS_SOURCE_REVIEW_RESULTS_20261017.md).

**2026-10-17 joint-source encoder probe completed:** Three fixed TRAIN event
windows pass frozen Whisper/EfficientAT/VideoPrism forwards and21-payload exact
fresh-process replay. Weights and prior artifacts are unchanged; measured audio
peak allocation200,990,208bytes, VideoPrism CPU host peak approximately4.14GB.
Physical time coordinates are saved without claiming local feature semantics.
No task fit or fusion gain. Next review TRAIN source labels/contact timing/grouping
and admit or reject a defensible comparison scope before full caching and learning.
Keep the original Attention/control policies and all product gates unchanged.
[Actual encoder results and limits](docs/pretrained_reuse/R2_GREATEST_HITS_ENCODER_RESULTS_20261017.md).

**2026-10-17 joint-source decode/tasks/input completed:** All409 development media
decode,17,932 windows pass, and23,707 TRAIN /8,875 validation questions rebuild
exactly. Three fixed TRAIN events produce18 exactly replayed input payloads:
native96kHz, independent16/32kHz paths, eight real RGB frames and actual PTS.
Exact cross-split raw/audio/video hashes are disjoint; physical offset, near
duplicates and cross-date actor/site identity remain unresolved. No encoder load,
gate fit, quality claim or product promotion. Next freeze and execute the same
three-event frozen-encoder smoke, then resolve complementary-task/intervention
admission before matched gate learning. The separate long unique-coverage question
remains unexecuted. [Actual results and limits](docs/pretrained_reuse/R2_GREATEST_HITS_DEVELOPMENT_RESULTS_20261017.md).

**2026-10-17 complete joint-source inventory/acquisition:** All733 official TRAIN
annotation files parsed. The prospective date partition remains308 TRAIN /101
development validation /324 quarantine. Native17/2/5-way fields support23,707 /
8,875 prospective questions, with no unseen validation class but marked label-mix
shift. All409 development AV files acquired/CRC checked (643MB). Complete CPU
decode is running; task construction and neural training are not claimed complete.
Observe the existing process, then inspect all window/content gates before building
the frozen task corpus. [Evidence and live-stage handles](docs/pretrained_reuse/R2_GREATEST_HITS_DEVELOPMENT_INVENTORY_20261017.md).

**2026-10-17 joint-source physical probe:** Greatest Hits official TRAIN labels
and one real AV clip were acquired with bounded version-checked ranges; full CPU
decode passed.98 annotation rows include54 complete/37 wholly missing/7 partial
labels. Official split IDs share55 timestamp prefixes, so source grouping remains
unresolved despite zero exact-ID overlap. Preserve missing labels and leave TEST
media/labels closed. Next inventory TRAIN labels/groups and establish task/alignment
before matched gate learning. No new fit or quality claim; AVC rights conflict is
REVIEW. [Actual source evidence](docs/pretrained_reuse/R2_JOINT_SOURCE_ADMISSION_20261017.md).

**2026-10-17 conditional AV interaction:** Add an opt-in residual with parallel
audio/video exchange and query/candidate-conditioned independent gates; preserve
the current Attention baseline and all trainer defaults. Four gate/exchange
ablations have CPU contract coverage. Instantiated reference total197,097,949
parameters fits200M. Frozen metadata confirms0joint AV observations in both TRAIN
and validation, so no joint learning or quality improvement is claimed. First
admit synchronized, complementary paired tasks with valid swap/missing/conflict
labels; then compare on a fixed learning protocol. The long unique-coverage
question remains separate and unexecuted. No audit or product promotion.
[Implementation, exact counts and limits](docs/pretrained_reuse/R2_CONDITIONAL_AV_INTERACTION_20261017.md).

**2026-10-18 AVA Active Speaker source screen:** Official train/validation
annotations provide per-face speaking/audibility labels and are described as
CC BY 4.0, making them a relevant candidate for bounded voice-to-face association.
Underlying YouTube media rights, local archive hashes and video/person split
integrity remain unresolved; no data was downloaded or used. Keep the candidate
`REVIEW`; this does not admit it as a both-required benchmark or justify gate
training. See `docs/pretrained_reuse/R2_AVA_ACTIVE_SPEAKER_SOURCE_SCREEN_20261018.md`.

**2026-10-17 full-extent comparison completed:** All8936 Image/456 Video caches,
32768 updates, exact best/final replay and saved-prediction audit PASS. Selected
Image/Video Accuracy improves, but Audio falls3.096pp and macro NLL1.630→1.674;
the predefined retention rule fails. Preserve this result without adoption or
geometry/seed search. Video's Accuracy interval includes0 and its NLL worsens.
Next preregister one matched32768-update unique-coverage comparison using the
already-frozen fresh TRAIN pool and original crop/control. The earlier coverage
comparison only measured2048 updates. Keep model/LR/selector/validation fixed;
record Video's unchanged256-scene coverage. This next comparison is not yet run.
Temporal/LSTM/xLSTM, joint/OOD, calibration and95–98% product gates remain open.
[Measured results](docs/pretrained_reuse/R2_NATIVE_FULL_EXTENT_RESULTS_20261016.md).

**2026-10-16 full-extent comparison registered:** Keep the original control's
sampling/model/long budget and replace only Image/Video center crop with one
288px full-extent bilinear/black-pad policy. Geometry tests precede real CPU
encoder smoke, exact original-feature parity, complete new caches and a separate
training comparison. Object-scale/padding trade-offs and all modality declines
remain visible; no geometry/seed search or sealed audit.
[Frozen protocol](docs/pretrained_reuse/R2_NATIVE_FULL_EXTENT_PROTOCOL_20261016.md).

**2026-10-16 positive-support comparison completed:** One fixed32,768-update run,
exact best/final/parent replay and full sampler/LR/metric audit PASS. Non-speech
balanced Accuracy50.00→54.11%, but selected macro NLL1.630→1.679 and Audio
61.27→45.06% fail the predefined retention rule. Preserve the failed policy;
do not adopt or search its ratios/seeds. Next isolate full-field-of-view resize/pad
against the original control, with new feature/output paths and unchanged IDs,
model/budget/selector. Object-scale and padding trade-offs must be measured.
This result does not establish a model-capacity ceiling or complete the product goal.
[Measured results](docs/pretrained_reuse/R2_NATIVE_SUPPORT_RESULTS_20261016.md).

**2026-10-16 positive-support experiment fixed:** Compare one TRAIN-only STARSS
support-floor policy against the completed32,768-update control. Replace292
negative questions with unused positives, retaining every existing positive and
all non-STARSS occurrence slots. Same6656 unique rows,262144 presentations, model,
LR horizon, selector and full validation. Bell support remains unresolved. Run a
separate smoke, full fit and exact replay; no crop/frame change in this comparison.
[Prospective protocol](docs/pretrained_reuse/R2_NATIVE_SUPPORT_PROTOCOL_20261016.md).

**2026-10-16 データ／学習パイプライン監査完了:** 消費した音声イベント5種類は正例0。
うち4種類は元TRAINの正例をsamplerが未使用だった。画像は中央75%、動画は中央66.67%の
視野だけを保持し、動画128→8フレームの最大間隔は0.76秒。全件の入力対応・特徴transport・
独立ラベル計算・実入力の蓄積勾配・保存評価集計では取り違えを確認しなかった。
次は正例被覆と視野保持を別々に検証し、その後に時間処理・regularizationへ進む。
既存runは固定。追加学習0、sealed audit未使用、製品目標未達。
[監査結果と次の順序](docs/pretrained_reuse/R2_NATIVE_PIPELINE_AUDIT_RESULTS_20261016.md)。

**2026-10-16 data/pipeline priority correction:** The user requests scrutiny of
data and the actual learning path. Audit source semantics, consumed label support,
feature transport, spatial/temporal information loss and loss/gradient equivalence
before the proposed regularization fit. Preserve all completed runs. A reproducible
cache is not evidence that the task's information survives preprocessing.
[Audit protocol](docs/pretrained_reuse/R2_NATIVE_PIPELINE_AUDIT_PROTOCOL_20261016.md).

**2026-10-16 long learning curve completed:** One fixed native attention run
finishes 32,768 updates, followed by exact full best/final replay. Within-run
2,048→32,768 macro Accuracy improves 32.83→45.70%, while NLL worsens 1.740→4.848;
validation NLL selects 8,192. Video TRAIN reaches 100%, but validation 37.25% and
temporal 33.5% show no clear late generalization. Short runs did miss continued
Accuracy gains; this does not demonstrate grokking or a capacity ceiling. Preserve
best/final and all history. Next isolate regularization under a fixed long budget
before combining data/architecture changes. No additional run, audit or promotion.
[All learning curves and limits](docs/pretrained_reuse/R2_NATIVE_LONG_BUDGET_RESULTS_20261016.md).


**2026-10-16 longer-training correction:** The user notes possible delayed
learning/grokking. Both coverage best checkpoints are at the final2,048 updates;
TRAIN is still weak. Prioritize one fixed32,768-update attention curve before
architectural changes. Preserve the same repeat pool, seed and model; freeze a
new full-horizon cosine schedule and evaluate complete consumed TRAIN and full
validation. No performance early stop or claim of a capacity ceiling. This new
prospective experiment supersedes the next-gradient-audit priority below.
[Fixed protocol](docs/pretrained_reuse/R2_NATIVE_LONG_BUDGET_PROTOCOL_20261016.md).


**2026-10-16 matched-budget unique coverage completed:** Both arms finish2,048
updates and exact full replay. Fresh nonVideo questions improve Image/Audio but
reduce Video37.50→32.75% (paired scene95% interval for difference−8.5 to−1.0pp).
Macro NLL improves, macro Accuracy does not; the predefined retention rule fails.
Keep both artifacts without adoption or a clean-plateau/capacity-ceiling claim.
Next use a fixed TRAIN-only gradient audit to distinguish shared-head conflict
and clipping from other optimization limits before a bounded readout change.
No automatic extra seed/budget, recurrence expansion, audit or product promotion.
[All metrics, consumption and limitations](docs/pretrained_reuse/R2_NATIVE_COVERAGE_RESULTS_20261016.md).

**2026-10-16 actual consumed TRAIN evaluation completed:** Five immutable
initial/endpoints, all6,656 questions and exact full replay. TRAIN quality is
also weak; this does not identify capacity versus optimization/coverage.
Separate43 common tasks from48 TRAIN-only/28 validation-only tasks. Next test
repeat versus unused-TRAIN replacement at equal2,048 updates/eight microbatches,
with exact source/task/option-count quotas, fresh initialization and all existing
validation gates. NonVideo can add6,144 unique rows; frozen Video has none unused.
This prospective comparison is not yet executed; all prior artifacts remain fixed.
[Measured reachability and limits](docs/pretrained_reuse/R2_NATIVE_CONSUMED_RESULTS_20261016.md).

**2026-10-16 matched-state batch comparison completed:** Both controls finish
1,024 updates with identical sample multiplicities, candidate permutations,
initialization and full validation. Best/final/contrast fresh replay is exact.
Macro Accuracy improves30.36%→31.61%, but macro NLL does not; consumed text
both-correct2/442→1/442 and video0/16 in both. Close this batching intervention
without adoption, seed retry or automatic budget extension. Next measure task
reachability on the actual consumed TRAIN rows, then choose one evidence-based
readout/coverage change. Preserve all temporal/joint/OOD and95–98% gates;
do not equate this failure with a capacity ceiling or expand xLSTM from it.
[Complete measured comparison](docs/pretrained_reuse/R2_NATIVE_MATCHED_BATCH_RESULTS_20261016.md).

**2026-10-16 fixed TRAIN state diagnostic completed:**1,250 exact-question/full-
candidate pairs pass full fresh replay with frozen initial/2048 weights. Both-
correct rates are0/442 consumed text pairs and1/16 consumed video pairs; image
sensitivity does not imply correct value readout. Preserve17 unpairable task
strata, including unavailable unconsumed video. No training or audit access.
The proposed matched-state grouping comparison was already executed in the
matched-state batch study above: both arms used the same 8,192 occurrence
multiset, initialization, architecture, loss, optimizer and 1,024-update budget;
all unpairable rows and full validation/retention scope were preserved. Its
macro NLL and text/video both-correct criteria did not support the intervention,
so do not repeat it. No xLSTM expansion, product promotion or R2 completion.
[Measured diagnostic and interpretation](docs/pretrained_reuse/R2_NATIVE_STATE_DEPENDENCE_RESULTS_20261016.md).

**2026-10-16 common native fits completed:** Attention baseline and residual
LSTM each finish2,048 updates on identical data/order, with exact full best/final
replay. Temporal-video endpoint Accuracy remains33.5% in both. LSTM's small
all-video gain does not establish a memory benefit; macro NLL/image regress.
Keep both research artifacts, adopt neither as a qualified product, and defer
xLSTM expansion. Next resolve native-value readout/unique-coverage limitations;
only then use temporal interventions and a nonrecurrent capacity control to
isolate memory. STARSS all-no predictions remain a failure despite83.38% Accuracy.
No audit, quantization, legacy artifact modification or R2 completion.
[Full results and evidence limits](docs/pretrained_reuse/R2_NATIVE_TEMPORAL_FIT_RESULTS_20261016.md).

**2026-10-16 temporal state extension:** User-authorized architecture flexibility
adds an optional LSTM residual candidate at197.291M total parameters. Native
feature/GPU/gradient/replay checks pass; the existing baseline is preserved.
This is not a task fit or quality result. Complete common supervision first,
then compare under fixed data/selection/retention rules. xLSTM remains untested.
[Actual execution and comparison policy](docs/pretrained_reuse/R2_TEMPORAL_STATE_RESULTS_20261016.md).

**2026-10-16 complete language/audio extraction:** All4,382 states/22,328 pairs
and7,410 native audio examples are persisted and pass full independent feature
recomputation with zero difference. Both audio paths are present for each example;
all1,135 distinct source/input pins pass. The prior disk-blocked output is preserved.
Next bind image/audio/video query features and freeze one shared fit contract,
then train the current readout. Zero optimizer updates here; R2 remains open.
[Actual execution and remaining coverage](docs/pretrained_reuse/R2_NATIVE_FULL_FEATURES_RESULTS_20261016.md).

**2026-10-16 strategy refresh, 14:35 JST:** Keep the 196.25M candidate and focus
the next work on common inputs/run contract, one shared-readout baseline fit,
then one evidence-directed repair. The readout still has zero updates. C: free
space now measures10.38GB; recheck each run's reserve instead of assuming the
old disk blocker persists. No new fit or quality result in this strategy review.
[Execution priorities and decision rules](docs/pretrained_reuse/STRATEGY_REVIEW_20261016.md).

**2026-10-16 input execution:** Lossless-token language windows and unified
speech/event preprocessing pass actual model smokes; all6 audio arrays reproduce
exactly in a fresh process. Full language caching stops before model loading at
the fixed disk-reserve gate and remains incomplete. Shared-readout fitting has
not started. [Evidence and restart conditions](docs/pretrained_reuse/R2_NATIVE_LANGUAGE_AUDIO_RESULTS_20261016.md).

**2026-10-16 common training preparation executed:** Persisted all256train/
200development-validation eight-frame token arrays; all456 historical hashes
and fresh file rechecks pass. Shared-source audit finds38 long text states,
speech-cache numerical mismatch and incomplete combined audio features. Resolve
these input gaps, then freeze common fit/retention rules; no model update, quality
claim or R2 completion. [Evidence and next unit](docs/pretrained_reuse/R2_NATIVE_TRAINING_PREPARATION_20261016.md).

**2026-10-16 native candidate integration measured:** The same original-pretrained
composition accepts native text/image/audio/video and supplied conditions/options,
with **196,254,874 actual parameters before quantization**. All 21 arrays and eight
JSON decisions reproduce exactly in a fresh process. Its new 889,345-parameter
readout has zero training updates; quality, text-free task mode, joint/temporal/OOD,
calibration and packaged deployment remain open. This closes the functional
integration unit, not R2. Next freeze native data/task/retention gates and learn
this readout before considering a further architecture change or compression.
[Measured inputs, resources and hashes](docs/pretrained_reuse/R2_NATIVE_CONDITIONAL_CANDIDATE_20261016.md) ·
[Current strategy](docs/pretrained_reuse/NEXT_STRATEGY_20261016.md).

**2026-10-16 matched supervision comparison:** Both1024-update fits complete,
with exact reload and all original source/input pins unchanged. Both endpoints
reach100% on new16 validation slices; no trained checkpoint retains all47
slices, so both selected steps remain0 and the auxiliary promising gate fails.
All three fixed weights pass536-question TRAIN replay. Fixed256-state transfer
and independent output audit complete: endpoints CE93.70% /relevance94.17%, with
minimum16-slice Accuracy50.00% /50.88% and an actual syntax slice at0% for both.
Close this comparison; no coefficient/budget sweep or promotion. Next define
one native candidate/task contract and measure it across independent native,
conditional, temporal and joint tasks, retaining the <=200M and95–98% gates.
[Completed fits](docs/pretrained_reuse/R2_CONDITION_RELEVANCE_FIT_RESULTS_20261016.md) ·
[Completed fixed transfer](docs/pretrained_reuse/R2_CONDITION_RELEVANCE_TRANSFER_RESULTS_20261016.md) ·
[Current strategy](docs/pretrained_reuse/NEXT_STRATEGY_20261016.md).

**2026-10-15 fixed native-image bridge comparison completed:** Architecture A
remains primary. The1024 semantic endpoint improves oracle41.80%→73.83% and
image-predicted41.11%→73.34% on256 observed TRAIN images, but count is0/256 even
with correct facts. All268 oracle errors select true-but-unrequested attributes;
54 native count errors are additionally masked by this downstream failure.
Full26-payload fresh replay and independent20480-prediction audit pass. Next define
one independent condition-to-property supervision/retention contract; no same-probe
prompt, field-mask or checkpoint search. The fit's frozen selector remains0.
No adoption, independent native-quality or four-modality95–98% claim. Complete
<=200M, native speech, screen/sensor, joint/temporal/OOD requirements remain open.
[Updated priorities and stopping rules](docs/pretrained_reuse/NEXT_STRATEGY_20261015.md) ·
[Actual image-bridge results](docs/pretrained_reuse/R2_SEMANTIC_NATIVE_RESULTS_20261015.md) ·
[Completed fit evidence](docs/pretrained_reuse/R2_DECISION_SEMANTIC_FIT_RESULTS_20261015.md).

The chronological entries below retain their historical scope. Use the current
strategy above for the next action, without changing any frozen experiment rule.

**2026-10-15 independent decision supervision actually generated:** Fixed 16,384
TRAIN / 3,072 selection / 4,096 development-transfer questions; 2,432 new state
classes, all candidate widths retained. Historical 7,612 binding and 3,328 numeric
classes excluded; exact legacy replay/validation preserved. All 13 files reproduce
byte-for-byte. Validation selection requires all 31 slice counts and retains
step 0 in macro-NLL minimization. No model load, optimizer step or audit access.
This completes the data/selection deliverable, not model quality. Next freeze one
semantic training comparison and its uncertainty runner, then connect successful
decisions to native temporal/joint inputs. Product goals remain unmet.
[Execution, hashes, tests and limitations](docs/pretrained_reuse/R2_DECISION_SUPERVISION_RESULTS_20261015.md).

**2026-10-15 representation/factorization screen actually executed:** Two frozen
checkpoints,32unused states/eight syntaxes/1,024questions/all40choices, no training.
DeBERTa34.08%→59.08% with factoring (state-paired95% change interval+21.58 to+28.91pp),
but weakest-family25% and retention fail; MiniLM32.42%→28.91%. Full fresh replays
match25payloads/model exactly. ExecutionPASS, research gatesFAIL, no adoption.
Close this fixed rendering screen. Next prepare joint conditional-relevance,
state-consistency, numerical and original-task retention contracts before any fit;
no prompt/temperature sweep or repeated old numeric probe. DeBERTa's195.37M
encoder-only speech scenario leaves4.63M before all heads and has unproven speech
quality. Whole <=200M/native four-modality95–98%/temporal/joint/OOD gates stay open.
[Full measurements and limits](docs/pretrained_reuse/R2_FACTORIZED_REPRESENTATION_RESULTS_20261015.md).

**2026-10-15 matched conditional-contrast comparison completed:** Two 512-update
fits use the same new states/replay/order and differ only in the auxiliary loss.
Focus+new-description Accuracy falls 23.14%→17.38% (paired 95% difference interval
[-7.324, -4.297] pp); complement+new-description falls 97.27%→96.39%.
Neither endpoint meets retention; both selectors fall back to step 0. The prior
KL512 remains a historical reference, not a generally qualified model. All 86
full-transfer replay files match. Saved-error attribution identifies wrong-field
selection with correct state values; it does not prove a model-capacity ceiling.
Close this loss comparison. Reassess conditional-language representations and
the whole <=200M allocation before another semantic fit, then connect native
perception and temporal/joint decisions. No promotion, audit access or quantization.
[Full execution, learning curves and limitations](docs/pretrained_reuse/R2_CONDITION_CONTRAST_RESULTS_20261015.md).

**2026-10-15 strategy synthesis and fixed teacher factorial completed:**
Display order and label assignment are separated on the same1536 TRAIN prompts.
Binding Accuracy is81.25/59.38/74.22/42.97% across00/10/01/11; numeric52.73–54.69%.
All768 historical endpoint logits and48 predeclared fresh replay outputs match.
Do not admit this teacher interface or sweep prompts/labels. Next freeze independent
conditional-contrast supervision and retention contracts, compare one training factor,
then connect the same decisions to native inputs and temporal/joint tasks.
R0–R8, <=200M before quantization, each-task95–98%, legacy artifacts and final audit
remain intact; product goals remain unmet.
[Actionable three-deliverable strategy](docs/pretrained_reuse/NEXT_STRATEGY_20261015.md) ·
[Complete diagnostic evidence](docs/pretrained_reuse/R2_TEACHER_LABEL_FACTORIAL_RESULTS_20261015.md).

**Active strategy, 2026-10-15:** The user requested a synthesis of accumulated
results and a new priority order. Preserve R0–R8 and all historical evidence;
prioritize conditional semantics, task-grounded temporal/multimodal data and a
compact Decision core. Close the failed six-coefficient and repeated small-scene
readout searches. Define a shared capability/data contract and one discriminating
comparison before further fitting; quality precedes quantization. The <=200M,
native four-modality, reusable-state/JSON and95–98% objectives remain unmet.
[Evidence, architecture budget, next experiments and stop rules](docs/pretrained_reuse/STRATEGY_20261015.md).

**2026-10-15 TRAIN-side semantic teacher screen executed:** Existing pinned Gemma4
BF16, no adapter/fitting/audit:1,152 prompts/64 TRAIN states, all40 binding and4/5
numeric choices retained. Explicit/switch98.44%, exclusion81.25%, reversed
exclusion42.97%, numeric54.69%; option-order agreement41.41%/51.95% fails both
fixed gates. Fresh predeclared36-row replay is exact. Do not distill this interface;
next separate semantic difficulty from option-label/position handling with a
prospective mechanism test. No teacher/model ceiling, native quality or product
admission; <=200M and four-modality95–98% remain open.
[Full metrics, controls, resources and hashes](docs/pretrained_reuse/R2_SEMANTIC_TEACHER_RESULTS_20261015.md).

**2026-10-15 retention-source coverage comparison executed:** One fixed512 fit,
same binding instructions/model/budget/loss/selector; replay changes from2048original
to1024original+1024curriculum. Curriculum Accuracy+1.46pp and NLL-.08939 versus
language512, with state-paired intervals on the improvement side. OriginalECE
still fails the unchanged gate; all trained checkpoints are ineligible, selected0.
Observed regression also trades focus-description Accuracy81.64%→76.27% against
complement-description6.45%→15.43%. Keep earlier KL512 reference; MIXED, no promotion.
Close this fixed ratio comparison and verify TRAIN-side semantic teachers before
another fit; do not optimize the same validation via ratio/temperature/step sweeps.
[Protocol, learning curves, all comparisons and artifacts](docs/pretrained_reuse/R2_BINDING_REPLAY_COVERAGE_RESULTS_20261015.md).

**2026-10-15 instruction-diversity comparison executed:** One fixed512 fit,
same states/options/replay/selection; only instruction text changed. On new
state-disjoint syntax, old512→new512 improves focus+descriptions31.15%→81.64%,
but complement+descriptions1.46%→6.45%. Both transfer-quality gates fail;
all trained checkpoints also fail old-task retention, so this trial selects0.
Preserve the earlier KL512 reference. Prioritize conditional meaning and explicit
retention-source coverage in separate comparisons; no size/step rescue or promotion.
[Learning curves, exact comparisons and artifacts](docs/pretrained_reuse/R2_BINDING_LANGUAGE_RESULTS_20261015.md).

**2026-10-15 independent conditional transfer measured:** Fixed parent and retained
candidate, 256 unused states, eight conditions, same 40 choices. Candidate is
100% on known wording, 49.22% with new question/option descriptions, and 9.47%
with exclusion instructions plus new descriptions. Both primary transfer gates
fail, despite improvement in some arms. No training, audit access or promotion.
Prioritize independent diverse conditional supervision; keep these observed
conditions as regression evidence and do not tune templates against them.
[Full metrics, controls and limits](docs/pretrained_reuse/R2_BINDING_TRANSFER_RESULTS_20261015.md).

**2026-10-15 fixed probability-retention comparison completed:** One seed-17
512-update fit preserves narrow binding Accuracy at 100% and passes all unchanged
old-task gates at step 512. Fresh selected scores and 12 payloads are exact.
Curriculum NLL still worsens versus the parent and extremum Accuracy is close to
the tolerance boundary. Freeze this research candidate; next measure independent
conditional/native transfer without further tuning on the current selector.
No transfer/audit evaluation or product promotion occurred.
[All curves, cost, bootstrap and limits](docs/pretrained_reuse/R2_BINDING_RETENTION_KL_RESULTS_20261015.md).

**2026-10-15 fixed conditional-binding fit completed:** One seed17/512-update
mixed continuation raises narrow semantic validation31.74%→100%, but every
trained checkpoint fails unchanged retention/calibration gates. Keep step0parent
and preserve final weights separately; no model promotion. Next investigate
retention of old option distributions before increasing capacity or steps.
This annotation-state task is not native Omni or unknown-language quality.
[All curves, selection, controls and limitations](docs/pretrained_reuse/R2_CONDITION_BINDING_RESULTS_20261015.md).

**2026-10-15 conditional-state diagnosis executed:** Fixed256TRAIN images,
same40typed options/four questions, no fit or validation/audit access. Correct
oracle facts yield27.05% overall despite100% within-correct-field accuracy;
all-four0/256 and changed-question pairs34/1536. Predicted facts yield23.93%.
Full fresh-process replay of22payloads is exact. Prioritize conditional binding
and an independent semantic-transfer data contract; this new interface/TRAIN
probe does not establish a universal encoder ceiling or product quality.
Close prompt/readout search on this probe; all four-modality goals remain open.
[Measured results, accounting correction and next action](docs/pretrained_reuse/R2_SEMANTIC_BRIDGE_RESULTS_20261015.md).

**2026-10-15 R2-M visual-state diagnosis actually completed:** Three fixed models,
all889development utterances, no training, full fresh-process replay pass. A
single training-derived latent preserves all5334joint answers; ordered joint
states are saturated and carry negligible variation to the readout. Video-only
48.01%→21.18% on cached-token reversal confirms order-sensitive readout use.
Close further capacity/epoch/auxiliary-loss exploration on this redundant task;
require complementary AV decisions with hard source-matched distractors before
fusion fitting. No new audio-only product or general Omni claim.
[Measured diagnosis and all controls](docs/pretrained_reuse/R2_LOMBARD_VISUAL_STATE_RESULTS_20261015.md).

**2026-10-15 R2-M temporal readout actually measured:** Same106355parameter head,
five fresh CPU arms×24epochs, full best/final replay and fixed controls pass.
Video-only ordered48.01% versus broadcast29.85%; ECE worsens and letters remain
6.97%. Joint99.53% does not establish semantic fusion: mismatched video changes
0/5334answers. No extra budget/capacity or promotion. Next distinguish fixed
visual/mask/parameterization contributions before further fusion fitting;
genuinely joint tasks and four-modality95–98% quality remain open.
[Measured results](docs/pretrained_reuse/R2_LOMBARD_TEMPORAL_RESULTS_20261015.md).
Bare-finger Audio admission stays held after notebook/author-link inspection;
reopen only with traceable primary channel mapping.
[Clarification and preserved limits](docs/pretrained_reuse/R2_BAREFINGER_PRIMARY_CLARIFICATION_20261015.md).

**2026-10-15 alternative native source actually read:** Direct CC BY4.0
bare-finger contact data provides500 trials and200 texture-shared photographs.
Full one-trial numeric/JPEG inspection and fresh seven-array replay pass; raw
audio/vibration column semantics conflict between paper and supplied reader.
No RGB video, independent split or quality admission. Resolve semantics and
shared-photo/specimen/calibration gates before fitting, retaining all product
and legacy constraints. [Measured evidence](docs/pretrained_reuse/R2_BAREFINGER_SOURCE_RESULTS_20261015.md).

**2026-10-15 source/task follow-up executed:** Real and silence audio reproduce
the same fixed-ASR text, leaving RoboMNIST counter/clock hypotheses unsettled.
New CC BY4.0 contact-material candidate VisTouch has24 source sessions /28min
unique window coverage, despite10,498 stretched overlapping clips. Supplied
validation is absent; video timing/specimen/media/source-identity gates remain
open. Audit all native sources before fitting; no archive access, quality, GPU
restart or full95–98%/<=200M Omni completion is claimed.
[Actual acoustic diagnostic](docs/pretrained_reuse/R2_ROBOMNIST_AUDIO_CUE_RESULTS_20261015.md) ·
[Complete metadata and access limits](docs/pretrained_reuse/R2_VISTOUCH_SOURCE_RESULTS_20261015.md).

**2026-10-26 R2 VisTouch Git payload recheck:** A standard shallow clone at
`liangnjupt/VisTouch@a569c7e572a78f232e819b652e0ebb718952e476` succeeds, but
the Git tree contains no raw audio/tactile/video (dataset has only `.gitkeep`,
zero LFS files, and no GitHub release). The pinned README directs users to the
external Baidu archive. Metadata confirms 24 source sessions and a grouped
3N/6N-train vs 9N-test split; the published test is already observed and cannot
serve as a blind audit. Dataset terms are declared CC-BY-4.0, code MIT. No media
or model was downloaded or evaluated. Keep the source as a research candidate;
Path A training awaits an authorized native-media route. See
[`R2 VisTouch Git recheck`](docs/pretrained_reuse/R2_VISTOUCH_GIT_RECHECK_20261026.md).

**2026-10-15 native frontend/container gate executed:** On one known-train pilot,
harmless container padding leaves full native decoded frames and frozen TinyCLIP
features exact, but changes both fixed length-only predictions. Fresh CPU model/
native replay passes; all resident83.1M parameters are counted. Conditional and
temporal quality, common clock, native corpus/split admission and joint benefit
remain open. Retain nuisance controls before fitting; no legacy GPU restart.
[Actual execution and bounds](docs/pretrained_reuse/R2_ROBOMNIST_CONTAINER_FEATURE_RESULTS_20261015.md).

**2026-10-15 RoboMNIST shortcut evidence measured:** Complete2,135-group metadata
diagnostic uses1,475train/660validation and preserves global-Rp grouping/all60
classes. Camera lengths alone predictactivity60 at65.45%; all-length fusion has
lower43.48% Accuracy despite improved NLL. No native/encoder quality or blind
claim. Next check container invariance and source-free decoded preprocessing;
keep length/no-input controls. Full Omni/95–98% gates remain open.
[Measured five conditions and replay](docs/pretrained_reuse/R2_ROBOMNIST_LENGTH_RESULTS_20261015.md).

**2026-10-15 RoboMNIST payload mechanics executed:** Fixed seven-input pilot
finishes in186s; fresh native decoding matches all27 arrays/92 pins. Actual CSI
timestamps start at0; WAV provides no common start clock, so synchronization
and corpus/split/complementarity admission remain open. Next verify immutable
collection timing and task contracts before joint fitting. No model/GT/GPU
work or restart; full Omni/JSON/95–98% requirements remain unmet.
[Actual native formats and replay](docs/pretrained_reuse/R2_ROBOMNIST_INPUT_RESULTS_20261015.md).

**2026-10-14 RoboMNIST full-directory stage executed:** Complete60-class
inventory and separate lexical replay pass:2,135 groups,32–44/class,17,080 files.
Source bytes/config/group accounting retained; no payload/timing/session/split
admission or model training. The8GiB disk-interrupted memory comparison remains
protected. Next bounded input-schema/sync evidence before any joint fit;
full native Omni/calibrated JSON/95–98% and OOD requirements remain open.
[Measured inventory and limits](docs/pretrained_reuse/R2_ROBOMNIST_DIRECTORY_RESULTS_20261014.md).

**2026-10-14 follow-up execution:** Separate4h native memory inference is incomplete
after8GiB disk-reserve stop;18,176 complete-arm +1,568 partial object-frames pass
saved-product geometry checks only. Full/fresh/proxy-quality gates remain closed.
New20.17M RTDETRv2-R18 passes CPU/fresh mechanics; no semantic adoption. Public
RoboMNIST version1 clears source-rights metadata and first-directory grouping,
not split/payload/complementarity admission. No new fitting or goal narrowing.
[Saved failure and next conditions](docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_INTERRUPTION_20261014.md) ·
[Actual detector audit](docs/pretrained_reuse/R1_RTDETR_V2_R18_CPU_RESULTS_20261014.md) ·
[Prospective joint-input source](docs/pretrained_reuse/R2_ROBOMNIST_SOURCE_20261014.md).

**2026-10-14 native mask-memory feasibility executed:** Public cache32 preserves
all raw outputs on the fixed2seed prefix and halves vision calls; independent
192-frame audit passes. The full8video/696seed comparison is not admitted because
projected9,945s exceeds7,200s. Keep the failed gate and full cohort; no quality,
fresh model replay or product claim. Any larger resource budget requires a new
recorded protocol without changing inputs/numerics or reviving rejected batching.
[Actual smoke, resources and unfinished comparison](docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_SMOKE_RESULTS_20261014.md).

**2026-10-14 original QA resolution comparison executed:** Both fresh heads finish
50 epochs / 500 updates on 79 fit / 20 selector TRAIN scenes. Best epoch 8 Accuracy
is 25% for both; final coarse/fine Accuracy is 45%/32.5%, with strong overfitting.
Full saved-output replay and independent metrics/accounting audit pass. Close this
resolution-only fit; require semantic identity/temporal evidence before another fit.
No blind result, product promotion or full Omni quality claim.
[All curves, controls, artifacts and limits](docs/pretrained_reuse/R2_FINE_QA_GROUPED_RESULTS_20261014.md).

**2026-10-14 fine retention/support executed:** FrozenTAPIR actual32×32tokens on
all99knowntrain clips finish in303s; coarse parity/fresh/full file audit and8clip
independent numerical reference pass. Same-rule support: hidden19→1,957pairs and
5→34scenes, but same-hidden gap8only30/endpoint0. Preserve all zero-support scenes;
require grouped source-free semantic/temporal evidence before another fit or QA
claim. No new model/training/quantization or release promotion.
[Measured finer inputs and support limits](docs/pretrained_reuse/R2_FINE_SPATIAL_AND_ROLE_RESULTS_20261014.md).

**2026-10-14 all99 train grid support executed:** Corpus preparation, full fresh
1,188-array replay and independent811,008-cell/2,111,587-pair audit pass. Only19
same-hidden pairs in5scenes; all99/zero-support scenes retained. Aggregate pair
count does not justify hidden-memory training; preserve fixed rule and require
actual finer small-object/time evidence plus diverse cue supervision. No fit,
development/audit scoring or model-capability ceiling claim.
[Measured support and next gate](docs/pretrained_reuse/R2_GRID_ROLE_CORPUS_RESULTS_20261014.md).

**2026-10-14 learned correspondence primitive rejected:** All8scene-held-out
folds/two arms finish512steps; trainappearanceAccuracy100% with held-outNLL worse
in everyfold (macro1.875 vs geometry0.536). Independent metrics/reload/sampler
audit passed. Only5,400proxy pairs of2.63M opportunities are scored; noQA/blind
claim. Close this fixed fit and require more diverse cue/temporal supervision.
[Executed learning curves](docs/pretrained_reuse/R2_MASK_PAIR_ASSOCIATION_RESULTS_20261014.md).

**2026-10-14 foreground inputs admitted; role support sparse:** All18,441 native
regions/8 knowntrain clips×32times retained;72-array fresh replay and independent
64-row numerical reference pass. Source-box hidden/cover co-support30times,20 in
oneclip; do not confuse correlated region count with diverse semantic supervision.
Next define input-only cue/time-memory supervision and grouped controls before fit.
[Actual foreground bank](docs/pretrained_reuse/R2_NATIVE_MASK_FEATURE_CORPUS_RESULTS_20261014.md) ·
[Training-only census](docs/pretrained_reuse/R2_NATIVE_ROLE_OPPORTUNITY_RESULTS_20261014.md).

**2026-10-14 native wider corpus admitted:** Unchanged native run finished all
256 times on eight known train clips;39 reused/217 new. Fresh fixed-frame replay
and independent complete geometry/NMS/RGB/hash audit passed. Next: training-only
semantic cue/cover-memory input/supervision, with no GT runtime seeds or quality claim.
[Executed full admission](docs/pretrained_reuse/R2_NATIVE_DYNAMIC_CORPUS_FULL_RESULTS_20261014.md).

**2026-10-14 masked frozen-feature inputs verified:** All1,070 proposals over
32 times from one train asset are retained; full9-array replay exact and fixed8-row
independent numerical reference exact. No fitting/quality claim or live-run change.
Admit the native wider corpus before semantic cue/cover-memory supervision.
[Actual measurements and remaining gates](docs/pretrained_reuse/R2_MASK_FEATURE_POOL_RESULTS_20261014.md).

**2026-10-14 independent native input audit executed:**59/256 captured rows
(57 new,2 reused) passed complete1,024-mask geometry/NMS/grid/source-RGB checks
on two known train assets. CPU-only auditor leaves the active extractor
unchanged; no full-corpus, object-identity, QA or95–98% quality claim.
[Prefix evidence and final admission gates](docs/pretrained_reuse/R2_NATIVE_DYNAMIC_CORPUS_PREFIX_AUDIT_20261014.md).

**2026-10-14 native dynamic input expansion running:** Native-wrapper smoke
passed with five arrays containing all1,024 candidates, hashes/freeze/RNG retained.
Eight known train clips
use unchanged32times; reuse39 frame records and infer217 new. Main is started,
not admitted/completed; require terminal full coverage, fresh replay and CPU
audit before semantic cue/cover supervision. No nonadmitted batch path used.
[Actual smoke and live-run gates](docs/pretrained_reuse/R2_NATIVE_DYNAMIC_CORPUS_SMOKE_20261014.md).

**2026-10-14 fixed chunk admission executed:** Chunks16/32/64 fail unchanged
score tolerance; every discrete output and fresh16-array replay match. Actual
head call counts confirm equal64-object work. Retain all arms and stop batch-width
search; broader semantic cue/cover input needs native reference or its own fixed
derived-input protocol, with no frozen-corpus substitution.
[Measured matrix and source-gate correction](docs/pretrained_reuse/R2_SAM2_CHUNK_MATRIX_RESULTS_20261014.md).

**2026-10-14 proposal-forward engineering comparison:** Same frozen SAM2/native
prefix64, alternating warmed equal-work windows: ~5.04× cached forward ratio.
All discrete outputs match, but score maxerror1.335e-5>predefined1e-5; prefix
nonadmission retained, no silent tolerance change/corpus substitution. Fresh
replay passed; freeze/guard/resource checks recorded. Require explicit new
chunk-size/derived-input admission before wider semantic cue/cover data work.
No decision-quality improvement or product latency claim.
[Measurements and failed numerical guard](docs/pretrained_reuse/R2_SAM2_BATCH_DECODER_RESULTS_20261014.md).

**2026-10-14 timed spatial relation executed:** All1,070 queries×32times×3policies
predicted before posthoc labels. Later group current-bbox proxy69.13% actual vs
37.24% static; hidden-cue→cover0/18 actual and cue bbox retention2/18. One correlated
train cue, not QA or a model ceiling. Keep failed comparison, unknowns and
frame0 self-trajectory sanity cases. Close nearest geometry; require semantic
cue-to-cover binding/identity memory before another fit or broader claim.
[All arms, source exceptions and exact replay](docs/pretrained_reuse/R2_TIMED_SPATIAL_RELATION_RESULTS_20261014.md).

**2026-10-14 time-specific mask tracking executed:** Every1,070 NMS candidate
from all32 times became an own-mask query at its original frame; full558-frame
tracking and all15-array fresh replay passed. No GT seed selection, filtering,
training or QA claim. Retain backwards predictions with explicit observation-time
flags; this is offline inference. Next measure input-only target memory and
container association before fitting. Full Omni/joint/OOD quality remains open.
[Actual resource and integrity evidence](docs/pretrained_reuse/R2_TIMED_MASK_TRACKING_RESULTS_20261014.md).

**2026-10-14 per-time discovery pilot:** All32 fixed times/32,768 raw masks completed
on one known train regression scene. Group bbox opportunity improves over static
frame0; a later visible small-item cue survives NMS. Hidden-box1/19 is neither
recognition quality nor a visibility-denominated detection ceiling. Keep all13
unavailable times and all candidates. Next bind input-only time-specific queries
and measure occlusion/container association; do not GT-seed a chosen mask or
extend the failed readout. Cache parity passed, but product latency and full
Omni quality remain unqualified. [Execution and limits](docs/pretrained_reuse/R2_DYNAMIC_DISCOVERY_RESULTS_20261014.md).

**2026-10-14 input-only association comparison:** All256 queries/time were
predicted before posthoc labels. Future same-group bbox rates: actual45.86%,
static41.05%, original fixed-track51.72%; actual trails fixed routing in all8clips.
One of24 game groups lacks an initial unique grid point; keep it in the report.
No nearest-descriptor reassignment adoption or same99-scene budget extension.
Target/prelude association, relationship semantics and scene generalization still
need evidence. These reused training geometry proxies do not meet Omni quality.
[Executed comparison and limits](docs/pretrained_reuse/R2_TRACK_ASSOCIATION_RESULTS_20261014.md).

**2026-10-14 temporal appearance along model tracks:** Numerical routing on eight
known training clips passed official1e-5 sampling parity and exact fresh replay
for all256points×32times. Out-of-frame points remain with explicit support flags.
No learning, label-based selection or quality improvement is claimed. Establish
target/cover association and conditional relationship generalization before a
new readout fit; complementary multimodal and full Omni gates remain open.
[Executed input primitive](docs/pretrained_reuse/R2_TRACK_APPEARANCE_RESULTS_20261014.md).

**2026-10-14 condition/state diagnosis executed:** Existing best6/final50 only,
all99train/198QA; no fitting or development scoring. Final train95.96% drops to
49.49% with swapped questions and40.40% with mismatched whole scene; changed-target
both-correct46/53→0/53 or3/53. Question information remains, but prior development
failure/overfitting remains. Exact original replay passed after preserving reference
numeric flags; failed attempts retained. Inspect input-only object/cover association
before another fit. [Evidence and limitations](docs/pretrained_reuse/R2_CONDITION_BINDING_RESULTS_20261014.md).

**2026-10-14 complete spatial-context comparison executed:** All356 frozen clips
and fresh replay passed integrity gates. Same232,385-parameter readout compared
initial-broadcast versus32-time appearance, seed17/50epochs/650updates each.
Both selected Accuracy38.91%, changed-target both-correct0/127 and all controls
change0answers; later train95–97% versus dev33–36% shows overfitting. Close this
readout/data repetition without promotion; condition/state binding and useful
native fusion remain open. [Results and all curves](docs/pretrained_reuse/R2_SPATIAL_READOUT_RESULTS_20261014.md).

**2026-10-14 supervised role/QA comparison executed:** Same139,521-parameter readout,
seed17/50epochs, auxiliary loss0 versus1; every99train/257development video retained.
Selected Accuracy38.91%→40.27%, but grouped intervals include zero. Changed-target
pairs both-correct0/127; motion-zero raises role-supervised Accuracy42.22%.
Both late curves overfit. Close this comparison without product promotion or more
repetition; test later appearance/relations with input-only semantics next.
[Actual results, source exceptions and all curves](docs/pretrained_reuse/R2_ROLE_SUPERVISED_QA_RESULTS_20261014.md).

**2026-10-14 per-time appearance executed:** Same frozen31.07M TAPIR preserves
whole-frame low/hi spatial maps at32 predetermined times×8known train clips.
Freeze/RNG/readback/repeat/fresh-reload gates passed, adding0parameters. Numerical
variation alone does not prove later-target identity or cover association.
Define supervised input-only semantics and fixed controls before a new fit;
full Omni quality and package gates remain open.
[Actual extraction and limitations](docs/pretrained_reuse/R2_TEMPORAL_APPEARANCE_20261014.md).
All99 existing train clips/198 questions also passed source-role and canonical
answer-remapping checks; groups are2/3/4, and13 hidden tracks begin after frame0.
This permits candidate loss supervision, not GT-selected inference inputs.
[Executed supervision coverage and corrected audit attempt](docs/pretrained_reuse/R2_ROLE_SUPERVISION_20261014.md).

**2026-10-14 mask-temporal primitive executed:** Fixed-grid pooling keeps209
unsupported masks; corrected mask-internal queries then tracked all696 candidates
and4,979 original train frames. Initial group opportunity24/24; post-hoc GT
association yields371/500 future point-in-box versus177/500 static. No quality
adoption: later-object appearance, input-only cover selection and actor semantics
must be established before a new fit. [Evidence/review/replay/limits](docs/pretrained_reuse/R2_MASK_TEMPORAL_RESULTS_20261014.md).

**2026-10-14 static semantic ranking closed:** Frozen TinyCLIP encoded696
unchanged NMS mask crops and passed freeze/repeat/reload/readback checks. Distinct
group coverage9/24 versus presence10/24, complete groups1/8 in both; no adoption.
Do not prompt/threshold sweep these known train outcomes. Temporal target
association and variable-group supervision still need a fixed input-only protocol.
[Actual comparison and retained artifacts](docs/pretrained_reuse/R2_TINYCLIP_MASK_RANK_20261014.md).

**2026-10-14 temporal source-cue gate:** Eight known training clips have explicit
grounded hidden/cover/group roles; group sizes are2/3/4 and one hidden track begins
at224.64 raw frames were reviewed without changing source cuts or model outputs.
Define input-only prelude association and actor-relative semantics before fitting;
source IDs/events supervise training only, and first-frame coverage is not QA
quality. [Source/frame evidence and unresolved gates](docs/pretrained_reuse/R2_PERCEPTION_SOURCE_CUES_20261014.md).

**2026-10-14 SAM2 initialization gate:** All8,192 proposals and model-produced-mask
seeding completed on the fixed8train clips; frozen-state/repeat/readback checks
passed. Initial annotated covering-object bbox opportunity8/8 in raw/NMS banks
falls to2/8 under unconditional top3 presence ranking. This is not hidden-item
identification or QA Accuracy. Validate target cue, game-time anchors and
actor-relative semantics before a query-dependent fit; no GT-match seeding or
threshold sweep. [Execution and limits](docs/pretrained_reuse/R1_SAM2_AUTO_INIT_20261014.md).

**2026-10-14 R1 object-mask/memory candidate:** Pinned SAM2.1 Tiny passed full strict
load and fixed train-only image-frame/video propagation on the local GPU, with
38,962,498 registered parameters,512 persistent coefficients and measured~695MB
peak Torch allocation. Reporting failures were preserved and corrected separately.
This is functional admission only; label-free automatic discovery, semantic
correspondence and Decision/joint quality still need evidence before a fit/adoption.
[Audit, rights, buffers and runtime](docs/pretrained_reuse/R1_SAM2_TINY_20261014.md).

**2026-10-14 fixed native AV study completed:** Native extraction5,367/5,367 and
audio/video/joint24-epoch comparisons actually completed. Selected macro slot
Accuracy97.94%/30.15%/97.09%; joint-minus-audio−0.844pp, grouped95% interval
[−1.498,−0.260], with worse NLL/Brier. Video best3 versus final24 shows overfitting;
video mismatch does not establish useful joint contribution. Retain all artifacts
and the narrow audio reference; do not extend/adopt pooled video fusion or promote
an Omni product. R2 complementary visual/AV and general conditional quality remain
open. Earlier pending snapshots below are historical.
[Executed comparison and remaining gates](docs/pretrained_reuse/R2_LOMBARD_GRID_RESULTS_20261014.md).

**2026-10-14 fixed native AV readout implemented:** Three-arm train-only GPU
smoke passed at about70MB peak Torch allocation; explicit complete-cache gates
prohibit using the running extraction's partial arrays. Same-budget full learning,
joint-benefit controls and final uncertainty/report audit remain pending.
[Smoke evidence and full-study gates](docs/pretrained_reuse/R2_LOMBARD_GRID_READOUT_SMOKE_20261014.md).

**2026-10-14 actual native extraction started:** Same fixed common AV splits,
original16-frame CPU VideoPrism and separate new output; do not use incomplete
caches. Pinned audio/video/joint native-slot protocol preserves every option,
including25 letters, with independent calibration and grouped uncertainty.
Final feature integrity and quality remain pending; no product promotion.
[Started execution and next comparison](docs/pretrained_reuse/R2_LOMBARD_GRID_VISUAL_FEATURES_20261014.md).

**2026-10-14 bounded GPU port:** Real full-token parity failed despite small
pooled differences and faster CUDA forwards. Gates remain unchanged; continue
native CPU extraction for the fixed AV data. No full port job or quality claim.
[Measurements and remaining numerical uncertainty](docs/pretrained_reuse/R1_VIDEOPRISM_TORCH_20261014.md).

**2026-10-14 native AV command source:** Full Lombard Grid archives and5,390
audio records were checked; four training MOVs are broken in both official copies.
New common-valid train4,080 preserves calibration398/validation889 byte-for-byte.
Compact frozen Whisper Encoder CPU/CUDA smokes passed. Next measure native
unimodal/joint command readouts; this scoped speech test does not close general
AV/sensor reasoning or the95–98% quality gates.
[Acquisition, media failures and fixed layout](docs/pretrained_reuse/R2_LOMBARD_GRID_MEDIA_20261014.md).

**2026-10-13 CLAP follow-up completed:** Fixed Clotho-AQA learning reached
joint65.18% versus EfficientAT66.96%. NLL/Brier point estimates worsen, grouped
intervals include no change, and later epochs overfit. Keep the existing reference;
conditional fusion/readout and genuinely joint audiovisual evaluation remain open.
[Measured comparison](docs/pretrained_reuse/R2_CLOTHO_CLAP_RESULTS_20261013.md).

**2026-10-13 conditional-audio task completed:** All856 licensed Clotho-AQA
development sources were acquired, decoded and checked with180,004 cross-split
waveform comparisons. Fixed24-epoch joint Accuracy66.96% versus question-only
64.82% is mixed: grouped intervals include no gain and NLL/Brier are slightly
worse. Matching audio contributes to the selected joint readout, while final24
overfits (train93.64%, validation65.54%). No promotion or epoch extension.
Keep all3,422 questions, source grouping and disagreements; this is development
evidence, not synchronized AV/sensor quality or the95–98% product target.
[Results, hashes, curves and remaining work](docs/pretrained_reuse/R2_CLOTHO_AQA_RESULTS_20261013.md).

**2026-10-13 additional audio coverage:** Completed a fixed128-uploader FSD50K
positive-only augmentation after verified selective acquisition and9,984 waveform
comparisons.24epochs selected epoch9; later NLL worsened while training CE fell.
Accuracy89.97%→89.37%, Brier worsened, bell recall remained0. Do not adopt this
candidate or extend its budget based on tiny aggregate NLL improvement.
[Results and limits](docs/pretrained_reuse/R2_FSD_POSITIVE_RESULTS_20261013.md).

**2026-10-13 conditional-text reference:** FLAN-T5 Small passed exact pretrained
loading, but fixed384-update last-block/all-decoder adaptations remain near
chance (about24% development Accuracy). Consumed-train replay is also poor;
removing tokenizer-unknown JSON braces does not resolve it. Close these measured
format/scope comparisons without adoption or budget extension. The model is
77M resident parameters and cannot simply replace text beside VideoPrism and
Whisper under200M. [Results](docs/pretrained_reuse/R2_FLAN_NO_BRACES_20261013.md).
The R1 candidate inventory now includes this reference and CAV-MAE Sync, with
load, rights and quality status separated; older entries retain their audit scope.

**2026-10-13 AV reference follow-up:** CAV-MAE Sync's official pretrained
checkpoint passed strict CPU loading and real-input checks. Fixed200-question
development correspondence scored32% with one frame and35% with eight; paired
intervals do not establish improvement. Audio-window reversal leaves NLL nearly
unchanged. Close this readout comparison; no product adoption or training-budget
extension follows. Weight rights remain REVIEW and semantic fusion is unmet.
See [eight-frame comparison](docs/pretrained_reuse/R2_CAV_MAE_SYNC_EIGHT_FRAMES_20261013.md)
and [temporal control](docs/pretrained_reuse/R2_CAV_MAE_SYNC_TEMPORAL_CONTROL_20261013.md).

> **Current user constraints (2026-10-12, clarified):** Build ONE compact,
> genuinely multimodal Decision model with at most **200M total resident
> parameters before quantization**, including **text, audio, video, image,
> screen-capture and sensor handling**. The model must support **native direct
> decisions from any supported input modality without making a text question
> compulsory**, as well as **fusion of simultaneously present modalities**.
> Target **95–98% Accuracy** on predeclared representative per-modality and
> fusion tasks, then quantize/recover the same model to **>=95%** on those tasks.
> These are aspirational and currently unachieved; "world understanding" does
> not have a meaningful universal percentage without a specified task battery.
> This clarification supersedes older "sensor optional" and text-first wording
> as a future design priority. Past measurements remain unchanged.

> **Status: architectural redirection proposal in Draft PR; R0 inventory and initial R1/R2 audits/probes are underway as of 2026-10-11. A shared frozen-feature candidate-scoring core now serves the text/image/audio/video Path A probe runners; typed immutable observation caching has been exercised on 1,920 audio assets, 1,536 CLEVR-4 images, eight reused CLEVRER scenes, and reusable state/query/candidate text features, with saved predictions rechecked after cache reload. A byte-bounded LRU now serves the real frozen CLEVRER CPU scorer with exact direct-path output parity and measured evictions. A separate expanded CLEVRER descriptive readout used 256 train and 200 validation scenes; it improved NLL/Brier but not Accuracy over the prior small readout on identical new validation IDs and showed early overfitting. The initial frozen VideoPrism + MiniLM binary reasoning readout reached macro Accuracy 0.519 on reused development validation scenes. A new content- and scene-disjoint 200-scene control found near-chance Accuracy for fixed video+text, same-data text-only and shuffled-video readouts; shuffling scene video left NLL/Brier essentially unchanged, so this setup shows no clear useful scene-specific video contribution. A full-coverage static CLEVR-4 VideoPrism probe used 8,424 unique train images and scored 87.9% on the reused 512-image selector and 88.1% on a one-time 1,595-image validation subset; a fixed-task runtime replay matched 128 saved decisions on 32 images, while the 458MB bundle and 4.26GB peak CPU working set miss practical lightweight packaging targets. This is synthetic image taxonomy performance, not video quality. These are bounded development diagnostics, not blind audits or general video claims. Frozen synthetic-English text, CLEVR-4 image, narrow audio keyword, video, and a one-time near-uniform Japanese zero-shot text diagnostic are measured; synchronized audio-video quality, deployment-ready packaging, explicit V-JEPA checkpoint rights, and full R1/R2 exits remain open.**
> This roadmap replaces the *future direction* in the previous `main` roadmap, not the factual record of completed experiments. It does **not** start, stop, modify, resume, or declare successful any training job; merge or close any open PR; change an existing config, model, dataset, or checkpoint; or authorize paid compute.
>
> **One-sentence mission:** Build a locally deployable, genuinely multimodal, calibrated **Decision model**, preferably **100–200M total resident model parameters**, by reusing *already learned* video/world and cross-modal representations with as little new data, training, and inference work as possible. Neither Gemma 4 E2B nor EmbeddingGemma 2 is a required product backbone.

## 1. Product contract and research priorities

### Product behavior

- Accept **audio, video, image, screen captures, sensor streams and text** as **equally valid observation sources**, alone or synchronized. Support **direct no-text Decision** from each non-text modality; do not treat speech/video/sensor as mere attachments to a mandatory text question.
- A shared observation/world state may produce calibrated decisions **without any question text**, or under an optional typed, spoken or written criterion, over stable candidate/action IDs: `choice`, yes/no and ordinal `score`. A text/question encoder is optional and must be excluded from no-text tests. Return **option probabilities** and supported abstention; no generative language Decoder is required for the Decision output.
- Reuse one encoded observation for **many questions** without rerunning a costly video tower whenever its state has not changed; represent time and modality/source identity when relevant.
- The persistent world-state cache is a means to reuse temporal evidence, **not a claim of a general, causal, action-conditioned world simulator**. Add recurrent state prediction or action-conditioned dynamics only if measured task requirements justify it.

### Constraints, with distinct status

| Constraint | Contract / status | How to verify |
| --- | --- | --- |
| Total inference parameters | **Target: 100–200M**, including every resident text/audio/visual tower, Lens, projector, embedding table, head, and adapter; <100M is a stretch goal. A larger system is permitted **only as an explicitly labeled research/reference baseline**, not a qualifying compact release. | Exact loaded `named_parameters` / deduped tensor accounting, not model names or paper-size approximations |
| High decision quality | Required, but no global percentage is meaningful without frozen tasks, labels, input lengths, and decision policy. The parallel 2026-10-06 product roadmap proposed **>=95% per text/image/audio/video task**; preserve this as a **stretch product target**, not a measured result or automatic criterion across arbitrary tasks. Define task-specific gates **before** evaluating a contender. | Held-out per-task quality + confidence intervals, not just a pooled score |
| Low retraining cost | Reuse publicly licensed pretrained checkpoints first; prefer feature extraction, cached probes and small readouts, then small adapters, then selective alignment, then model merging. No full world-model pretraining in the default path. | GPU-hours, energy/cost where measurable, trainable parameter count **and** backward FLOPs / peak memory |
| Local execution | PC validation first, iOS/ARM target as hardware becomes available. No cloud request must be required in the final inference path. | Package bytes, peak resident memory, warm/cold latency p50/p95, battery/energy where measurable, offline functional test |
| Deployment packaging | **Initial engineering target <=200 MB** including all model files, scales, higher-precision exceptions, tokenizers, lenses and optional adapter; aspirational <=100 MB. This may require low-bit storage and is **not** guaranteed by meeting the parameter target. | Serialized artifact size and exact runtime load |
| Development resources | Available historical reference: RTX 3080 Laptop 16 GiB VRAM, 32 GiB host RAM; target isolated CPU tests and bounded GPU runs. The old <=US$15 rented-compute budget is a *planning ceiling*, not an authorization to spend. | Before each GPU experiment record device, driver, software stack, wall-clock cap, samples, and cost |
| Reproducibility / rights | Immutable checkpoint/processor revisions, SHA-256 for selected inputs/artifacts, rights-aware data manifests, sealed evaluation, reproducible preprocessing and seeds. | Fail-closed manifest audit, split leakage checks and saved run ledger |

**Optimization order:** (1) usable quality on the required tasks, (2) zero/minimal training for comparable quality, (3) parameter/byte/latency feasibility, (4) stronger sharing/merging **only where it improves the measured Pareto frontier**. Never promote a smaller model solely because it is smaller.

**User clarification (2026-10-11):** 100–200M describes the underlying model's
parameter count before quantization, including all resident components. The
deployment plan then quantizes that model; the provisional <=200 MB target
applies to the resulting deployment package, not to its FP32 reference weights.
Quantization normally changes representation precision, not parameter count.
The measured 170.5M / 687.3 MB FP32 reference therefore meets the provisional
parameter range while leaving quantized packaging unmeasured. Its FP32 size
alone is not grounds to narrow the intended Decision tasks or reject its model
capacity. Quality, calibration and actual runtime after quantization still need
measurement; the ~182 MB INT8 envelope remains an estimate only.

## 2. Historical evidence to retain — not the new architecture

Completed on legacy `main` as documented in [`docs/DURABLE_TEACHER.md`](docs/DURABLE_TEACHER.md):

- Pinned Gemma 4 E2B Decision LoRA Teacher; 256 optimizer steps on a frozen 4,859-record train set, 2,901 validation and 2,917 final held-out evaluation examples.
- Reported overall held-out accuracy **0.7343**, NLL **0.6380**, Brier **0.3344**, ECE **0.0344**. By modality: text **0.6723**, image **0.5703**, audio **0.9492**, video **0.4176**. These are **dataset-specific historical measurements**, not cross-model guarantees.
- Original video data represent a small number of independent clips; multiple decisions from one clip are not independent scenes. [`docs/DATASETS.md`](docs/DATASETS.md) retains licensing/source restrictions.
- The historical teacher is useful as an **optional, separately measured reference or option-probability source**, not a required training objective. The unsuccessful embedding-level self-distillation attempt motivates an ablation against geometry imitation; it is **not proven** that the loss alone caused the failure.
- Keep pinned checkpoints, manifests, test fixtures and prior logs byte-identical. Do not re-use repeatedly inspected old test sets for model selection; define a new sealed final audit for the new research sequence.

**Parallel work: as checked 2026-10-10.** GitHub has open PRs **#9–#14**, including Teacher-quality work and EmbeddingGemma ternary/QAT work, several in draft/stacked branches. A separate `codex/roadmap-compression-video-specialization` branch describes another compression/video program. Their results, historical run locks and ongoing processes are **not canceled or reinterpreted** by this document. No auto-merge, auto-close, checkout/overwrite of live worktrees, or stop/resume decision is authorized. Evaluate each branch independently before deciding to archive, merge, or repurpose its code. Especially do not claim exact optimizer resume from a weights-only warm start.

## 3. Competing architecture paths (not three compulsory stages)

### Path A — Frozen modular reuse (**mandatory low-cost baseline**)

Use real, available pretrained visual/video, text and audio encoders, plus tiny readout/fusion modules. Extract and cache modality-specific **token or intermediate features**; preserve temporal positions. A small conditional candidate scorer consumes cached features.

- Candidate visual/video: **V-JEPA 2.1 ViT-B**, pretrained and already distilled from a larger video model. Source-derived inventory is about **86.8M** including separate image/video patch embeddings; **recount the exact downloaded checkpoint and remove training-only predictor from deployment tally**.
- Candidate text: a small multilingual pretrained encoder (e.g., Ruri-v3-30m as a *candidate*, not a verified plug-and-play choice), or a compatible already-aligned text tower. Check Japanese and English performance before selection.
- Candidate audio: a small pretrained environmental-audio model (e.g., EfficientAT) and optionally a distinct ASR reference. **Audio event understanding is not speech comprehension.**
- For the initial reference, keep independently trained towers rather than pretending they already share weights. Target combined loaded model <=200M when feasible. This path may already satisfy the product constraints and must remain a valid winner.

**Advantage:** no new large-scale pretraining; features can be cached; clear isolation of modality-specific errors.

**Cost:** more than one resident backbone, and fusion may not discover genuinely joint signals without paired examples.

### Path B — Shared pretrained video backbone + lightweight modality Lenses (**primary research candidate, conditional**)

Reuse the same video-trained Transformer weights for image/video and, **only if feasible**, audio/text/sensor tokens via small learned or already pretrained input Lenses. Keep separate modality/time/position metadata and optionally modality-specific side information. Preserve the video representation as an explicit quality constraint.

- Start with **frozen** video backbone; do not average unrelated backbone weights.
- Prefer reusing **actual released Lens weights together with their verified backbone** when possible. ViT-Lens demonstrates an approach, **not pre-validated compatibility with V-JEPA**.
- Probe lightweight audio/text tokens with verified preprocessing and exact model revisions. If a Lens precedes a frozen backbone, training it still requires **backpropagation through the frozen backbone to the input**, thus may require costly activations. Compare this cost against Path A feature caching; small trainable-parameter count does **not** imply cheap optimization.
- Do **not** require raw multilingual text to be relearned from scratch through randomly initialized embeddings. Keep a small pretrained language encoder as a side path or revert to Path A if needed.
- Do not force complete latent equality. Evaluate shared event information and modality-specific information separately.

**Advantage:** potentially fewer resident Transformer blocks and real capacity sharing.

**Failure modes:** negative transfer, modality token distribution shift, video forgetting, expensive input-side Lens training, language capacity loss, excessive attention cost.

### Path C — Weight-aligned model merging (**research option, not default**)

Investigate Model Soups/task vectors for common-initialization models; permutation/basis alignment or FS-Merge-type approaches for verified compatible different initializations. SSAM-like methods have additional common-base assumptions.

- **Prohibited shortcut:** raw element-wise averaging of independently pretrained V-JEPA and an unrelated audio/text model merely because they encode similar events.
- Produce a tensor-level compatibility report first: names, shapes, depth/width, attention and FFN layout, nonlinearities, normalization, RoPE/position schemes, patch/token embeddings, tied weights and checkpoint ancestry.
- Compare each source on its own tasks, the candidate merge **before and after alignment**, and an independent Path A baseline. Model merging can require data/optimization; account for it.
- If alignment is not justified by tests or compute budget, **skip Path C**, not the whole product.

**Advantage:** may remove redundant resident towers without full retraining.

**Failure modes:** weight-coordinate mismatch, destructive interference, loss of temporal or unimodal information, hidden training costs.

### Architecture selection rule

Run **A first**, then attempt **B only if expected savings/benefits are measurable**. Treat **C as optional** after proven tensor/functional compatibility. If A meets quality and on-device budgets and B/C do not improve quality-cost trade-offs, **ship A**. A single shared Transformer is a *research hypothesis*, not a product requirement worth degrading performance for.

## 4. Interfaces and data contract to design before model-specific code

### 4.1 Observation features

Proposed versioned internal structure (not implemented in the present `DecisionExample` schema):

- `observation_id`, `source_id`, `source_revision`, `modality`, `time_start/end` or sample timestamps; input/media references; checksum; processing version; availability/missingness; rights/provenance.
- Encoded token array with sequence length and hidden dimension, valid-token mask, modality/time/space location, model revision and precision metadata.
- Cross-observation state/cache containing feature tokens or selected slots with explicit history window, invalidation policy, and the **exact** input checkpoint/preprocessor identity.

`World state` initially means **reusable, time-aware pretrained observation features**, not a trained latent dynamics engine. Start without compulsory 32/64 recurrent slots; compare full tokens, intermediate-layer selections, pooled features, and 16/32/64-slot compression on **held-out tasks**. Do not call any slot count sufficient without evidence.

### 4.2 Typed decision output

- Stable string option IDs and exact ordered option text; variable-size candidate set and masking; `choice` and `noul` probabilities, ordinal `score` distribution/expected score; consistent semantics for abstain if enabled.
- The scoring head is replaceable and intentionally small. Probability calibration is fitted on a **calibration-only** split; report quality and coverage at a specified risk threshold separately.
- Teacher and student token IDs are not interchangeable. Optional teacher training targets are keyed by **actual option identity and order**, with detached probabilities/logits and source provenance.

### 4.3 Cache validity and information security

- Feature caches must include the full key: model/code revision, checkpoint hash, preprocessing, frame selection, tokenizer, input hash, modality/time metadata and dtype. Never cache arbitrary mixtures under a single key.
- Caches from validation/test examples may be used only for **read-only evaluation**, never fitted into the training pipeline.
- Do not export copyrighted media or private data with the feature cache. New synchronized audiovisual/sensor corpora require component-level rights review; existing OneJev aggregate remains REVIEW/excluded.

### 4.4 Software/runtime isolation

The current `pyproject.toml` ML extra targets the **legacy Gemma stack** (Torch 2.6.x, Transformers 5.6.x). Candidate V-JEPA, Lens, TensorFlow VATT and mobile runtimes may require different dependencies. Maintain **separate lockfiles/env profiles** until demonstrated compatible; never silently replace the active Teacher/QAT environment. Validate real model load, forward and checkpoint hashes on the intended device; a config-only load is not sufficient.

## 5. Data and measurement protocol

### 5.1 Data groups and leakage

- Retain historical data/license tools. Create a distinct new experiment namespace and **new sealed test set**; select model, layer, feature extraction, frame policy, compression and all thresholds using train/validation only.
- Group visual/video decisions by original clip/scene and audio by source event/speaker as appropriate; count **unique independent scenes**, not questions, in bootstrap uncertainty and minimum-sample declarations.
- Initial smoke may reuse approved, license-audited existing text/Clevr-4/Speech Commands/CLEVRER data in train/validation. These *single-media* datasets do **not** establish audio-video joint alignment. Add synchronized paired data only when a verified need, license and split exist.
- Add difficult counterfactual **evaluation**: frame-order swap, source-video same/audio swapped, missing modality, irrelevant cues, option-order permutations, unseen candidate descriptions, unseen scene conditions, delayed new questions over the same cached observation.
- Use labels/ground truth whenever available. Compare an LLM Teacher only as optional ablation, disclose noisy synthetic labels, and preserve disagreements for adjudication.

### 5.2 Core outcome metrics

- Per-task and per-modality **accuracy / macro-F1 / Brier / NLL / ECE**; separately report abstention coverage versus error risk; no blanket 95% claim.
- Temporal and world-state tests: time-order discrimination, event sequence, object persistence, long-gap recall, whether new questions can be answered from the same cached observation, and accuracy after state-slot compression.
- Genuine multimodal gain: paired versus shuffled signals and unimodal baselines, not superficial similarity.
- Efficiency: unique train examples, GPU-hours/total cost, peak allocated and resident RAM/VRAM, actual checkpoint/package bytes, activation memory, backward cost, input tokens/frame policy, warm/cold CPU/GPU/iOS latency p50/p95 and where available energy.
- Every performance figure states **HW, clock/power/thermal policy, backend, precision, batch=1 unless explicitly changed, inputs, warmup, repeats and sample counts**. Benchmarks on disparate hardware are not directly comparable.

### 5.3 Initial pre-registered acceptance hypotheses (proposal, not measurements)

**H1 — reusable latent quality.** A frozen pretrained temporal encoder plus small readout captures specified decisions on unseen scenes without requiring new video foundation pretraining.

**H2 — shared-backbone savings.** Path B maintains each predeclared relevant task within **2 absolute percentage points** of Path A, with a paired/grouped uncertainty analysis, **and** reduces at least one measured deployment cost (package bytes, peak memory or p95 runtime) by **>=20%**, while respecting <=200M parameters. Other costs must not regress beyond their individually specified tolerances. These are **initial experiment gates**, adjustable *before* looking at outcomes, not claims of achieved performance.

**H3 — no destructive merging.** Path C preserves temporal and individual modality scores within the declared noninferiority bounds. A merge which reduces checkpoint bytes but harms joint/time-conditioned decisions fails.

**H4 — optional persistent memory.** Adding cache/state slots beats a no-memory/pooled baseline on delayed-query and missing-observation conditions by a predeclared margin, without violating memory/latency limits.

**T — minimal test.** First a deterministic 8–32-example smoke for data/forward/shape/numerics; then at least **200 independent scenes/clips when available and legally usable** for the chosen temporal paired validation. This is a **provisional floor, not a power calculation**. Grouped bootstrap and confidence intervals must be computed; use the observed event variance and desired detectable difference for formal required sample size. Preserve a sealed final set untouched by hyperparameter selection. Initial screening can be run with one seed; repeat selected comparisons with multiple seeds.

**D — decision policy.** `PASS` when all pre-registered quality, cost, provenance and runtime checks hold, including the lower confidence bound on noninferiority for quality where required. `FAIL` if a hard bound is violated with sufficient evidence. `UNCERTAIN` if sample coverage, intervals, compatibility, or device measurements cannot support a claim. Do not convert an uncertain result into a win by selectively changing task weights.

**C — counter-hypotheses.** Frozen features may lack task-relevant information; feature extraction could destroy temporal structure; alignment might suppress modality-specific details; shared token growth could make B slower than A; merging may invalidate weights; an uncalibrated head may inflate confidence.

**U — uncertainty.** Record dataset sampling/label noise, paired-scene clustering, random seed, synchronization error, video frame selection, numerical dtype/quantization, backend/clock/thermal drift. Report interval method and effective independent sample count; only compute combined standard uncertainty `u_c` and expansion factor `k` when its component estimates are actually available.

## 6. Workstream phases: exit gates determine the next phase

### R0 — Evidence preservation and design lock (**first action**)

- [x] Archive a machine-readable inventory of `main`, current independent branch/PR revisions, benchmark/data manifest hashes and experiment checkpoint hashes without modifying their bytes (`docs/pretrained_reuse/R0_INVENTORY.json`; 2026-10-10 snapshot).
- [x] Reconcile the current remote PR head/base states and local worktree/process snapshot; preserve all independent branches and runs. Read-only contribution/dependency review for PRs #9–#14: `docs/pretrained_reuse/R0_OPEN_PR_REVIEW_20261011.md`. All remain open and unchanged; CI for PRs #13/#14 was not verifiable from the unauthenticated status lookup.
- [x] Refresh the live PR/worktree/process/storage snapshot without modifying existing experiments (`docs/pretrained_reuse/R0_LIVE_RECHECK_20261028.md` and `.json`). The 2026-10-28 snapshot inventoried 28 PRs and 84 worktrees. A read-only follow-up on 2026-10-29 confirms PR #15 remains Draft/Open at the current branch head, 28 PRs (#9–#36) remain open, and 84 worktrees are present; see `docs/pretrained_reuse/R0_LIVE_AUDIT_20261029_1204JST.md` and `.json`. No experiment was interrupted or changed. This refresh does not close the open device/task-criteria gate below.
- [x] Specify provisional forward-looking per-task accuracy, calibration, coverage, confidence-interval, and candidate-run local resource criteria without retroactively changing prior results. See [`docs/pretrained_reuse/R0_QUANTITATIVE_ACCEPTANCE_POLICY_20261031.md`](docs/pretrained_reuse/R0_QUANTITATIVE_ACCEPTANCE_POLICY_20261031.md).
- [ ] Select the end-user deployment device, latency/energy ceiling, and complete approved task battery before product promotion. The user-confirmed engineering reference remains the local Windows PC with RTX 3080 Laptop GPU (16 GiB), CPU fallback, no cloud compute, 100–200M pre-quantization parameters, <=200 MB only after measured quantized export/reload parity, and >=95% accuracy per defined task (98% stretch). These remain engineering targets, not release passes. See [`docs/pretrained_reuse/R0_DEVICE_AND_QUALITY_TARGETS_20261031.md`](docs/pretrained_reuse/R0_DEVICE_AND_QUALITY_TARGETS_20261031.md), reconciled from the user-confirmed 2026-10-28 record at `codex/r2-audio-vibration-source-20261027` commit `87265cf2c59608b806e4975c17f9a143dbbfe0ab`.
- [x] Publish candidate matrix, evaluation plan and separate environment contracts (`docs/pretrained_reuse/R0_DECISION_MATRIX_20261011.md`). Device, no-cloud rule, provisional parameter/package budgets and per-task accuracy aspiration are recorded for engineering; forward-looking provisional calibration/resource criteria are tracked separately, while product release thresholds and task battery remain open.

**Exit:** frozen legacy baseline and written constraints; no historical evidence altered.

### R1 — Pretrained checkpoint feasibility audit (**no training**)

- [x] Verify V-JEPA 2.1-B checkpoint URL, source commit, SHA, encoder/predictor split, strict EMA load, image/video forward behavior, tokens and parameters; see `docs/pretrained_reuse/R1_AUDIT.md`. **That exact checkpoint remains REVIEW.** A separate pinned official Google VideoPrism-B checkpoint passes the explicit rights and local-load gates: 114,365,184 parameters, 8-frame finite forward, CC-BY-4.0 stated for released weights by upstream and Apache-2.0 on the official Hub card. A matched CLEVRER development probe is recorded in R2; independent task quality and package fit remain unresolved.
- [x] Inspect ViT-Lens released backbone/Lens pairing, PE Core tiny/small/base, EfficientAT, small Japanese/English text checkpoints, VATT-MA and any relevant MJEPA *released* checkpoint. Mark `code only` / `pretrained and accessible` / `unknown` distinctly; see `docs/pretrained_reuse/R1_RELEASE_LICENSE_FOLLOWUP_20261011.md` and the source-of-truth inventory. PE asset access/implementation rights and VATT access remain REVIEW/UNKNOWN as recorded.
- [x] Load and hash Ruri v3-30m as one small Japanese text candidate; a single zero-shot JamC-QA-V2 dev diagnostic was near uniform chance. It is not a Decision-head result, has unresolved training-data contamination risk, and JamC-QA remains `REVIEW` under this repository's license policy; see `docs/pretrained_reuse/R2_PATH_A_PROBES.md`. This does not close the broader language-candidate audit.
- [x] Test the selected compact local candidate forwards for preprocessing, shape, parameter count and memory. VideoPrism-B's official loader and pinned checkpoint were exercised on a deterministic 8-frame input; ViT-Lens original-backbone execution remains untested and is outside the initial small model budget.
- [x] Create compact source-of-truth inventory with `passed_load`, `license_allowed`, `has_real_weights`, `runtime_supported`, parameter tensor bytes, source revision and primary citations in `docs/pretrained_reuse/R1_CANDIDATE_INVENTORY.json`. The minimum executable/legal video-candidate gate is now met by VideoPrism-B; task quality, model package fit and broader candidate screening remain open. V-JEPA 2.1 itself stays REVIEW.
- [x] Screen the small sensor checkpoint MOMENT-1-small as a frozen feature-source candidate. Its pinned MIT checkpoint passed a local 128-sample × 6-channel CUDA embedding smoke; no task quality was measured. UCI HAR was not used for model selection because MOMENT pretraining includes UCR/UEA and the available official sources do not identify every constituent dataset, leaving overlap UNKNOWN. See `docs/pretrained_reuse/R1_MOMENT_SENSOR_20261011.md` and the candidate inventory. This does not close the sensor quality or deployment gate.

**Exit:** at least one executable, legally usable video candidate and one viable small text/audio path; otherwise pause architecture selection.

### R2 — Frozen modular baseline A (**main low-compute milestone**)

**2026-10-29 R2-M OpenPack source screen:** The pinned official OpenPack
release is a relevant video/wearable-sensor work-activity source, but its
non-RGB dataset is CC BY-NC-SA 4.0 and its RGB release is restricted to
academic research with redistribution prohibited. The official challenge
split also shares participants across train and validation. Keep it
`REVIEW_NOT_ADMITTED` for product-directed work; no payload was fetched. The
preprocessed IMU archive is 515 MB, above current local free space. See the
[pinned source screen](docs/pretrained_reuse/R2_OPENPACK_SOURCE_SCREEN_20261029.md).

**2026-10-29 R2-M XRFv2 Plus screen:** The official Kaggle API reports
`licenseName=Unknown` for the 210.8 GB data release. Zenodo CC BY metadata
covers the attached paper; the project repository's MIT license covers code,
not demonstrably the dataset. Keep the source unadmitted until data rights and
grouped splits are verified; both-input necessity is also unknown. No payload
or annotations were accessed. See the [source screen](docs/pretrained_reuse/R2_XRFV2_PLUS_SOURCE_SCREEN_20261029.md)
and [machine-readable evidence](manifests/pretrained_reuse/xrfv2-plus-source-screen-20261029.json).

**2026-10-29 R2-M MM-Motion source screen:** Although its modalities and
motion-risk task are relevant, the upstream repository's signed agreement
conflicts with the Zenodo CC BY metadata and restricts use to non-commercial
research, forbids diagnostic use and redistribution, and requires a signature.
Do not sign or use the source for product-directed fitting without explicit
rights clarification. Its subject-level archive count does not establish an
official grouped split, and both-input necessity remains unverified. No media,
labels, or annotation payload was accessed. See the [pinned source screen](docs/pretrained_reuse/R2_MM_MOTION_SOURCE_RIGHTS_SCREEN_20261029.md)
and [machine-readable evidence](manifests/pretrained_reuse/mm-motion-source-screen-20261029.json).

**2026-10-29 EgoViz-120 metadata split preflight:** A deterministic 144/36
session-group train/development split covers all 12 publisher categories, with
zero session/sequence overlap. It is not an admitted task: participant
independence, generated-label quality, raw video/IMU synchronization, and
both-input necessity remain unknown; local storage is below a safe MCAP
download reserve. See the [metadata-only preflight](docs/pretrained_reuse/R2_EGOVIZ120_GROUP_SPLIT_PREFLIGHT_20261029.md).

**2026-10-31 GrandTour source/task screen:** The official publisher says the
dataset is CC BY-SA 4.0 and includes synchronized camera, IMU, LiDAR,
proprioception, and RTK-GPS, but the Hugging Face data card at its pinned
revision labels the dataset `MIT`. The 2.11 TB total, unresolved license
metadata conflict, and lack of a frozen mission-grouped Decision protocol
keep it `REVIEW_NOT_ADMITTED`; do not fetch payloads or use hidden COMFORT test
labels. A bounded visual-inertial option-selection pilot remains a candidate
only after rights, per-mission size/hash inventory, and grouped split gates
pass. See the [source/task screen](docs/pretrained_reuse/R2_GRANDTOUR_SOURCE_SCREEN_20261031.md)
and [evidence manifest](docs/pretrained_reuse/R2_GRANDTOUR_SOURCE_SCREEN_20261031.json).

**2026-10-29 R2-M Eidon video/IMU match diagnostic:** Frozen VideoPrism and
quaternion-only features were extracted for 120 train and 24 development
recordings after applying a contributor/task-group quarantine for malformed
quaternions. A pooled readout and a 2,000-update LSTM each reached 7/24
development matches; the long curves overfit, and reversing sensor time did
not reduce LSTM accuracy. The set has only 3 development contributors, so this
does not pass the R2-M both-input-necessary or quality gate. No sealed audit was
read; audio/image/text and the product gate remain open. See the
[reproducible diagnostic](docs/pretrained_reuse/R2_EIDON_TRACKER_POV_MATCH_20261029.md).

**MEVA metadata acquisition (2026-10-28):** The official annotation Git
repository is pinned, but the C: checkout is incomplete and consumes most
available disk; no video clips or model fit were started. The source separates
training-level and evaluation-level annotations. Rights for the specific
video payload and a clip-disjoint train/validation split remain gates. See
[`R2_MEVA_GIT_METADATA_ACQUISITION_20261028.md`](docs/pretrained_reuse/R2_MEVA_GIT_METADATA_ACQUISITION_20261028.md).
WEAR is a synchronized video/IMU alternative, but its official CC BY-NC-SA
license excludes it from this product-directed training path; no data were
downloaded ([source screen](docs/pretrained_reuse/R2_WEAR_SOURCE_SCREEN_20261028.md)).
EgoSight-Sample-500H's Hub card reports CC BY 4.0 and synchronized video/IMU,
but unauthenticated metadata returned 401 and consent/split details remain
unverified; it stays REVIEW ([source screen](docs/pretrained_reuse/R2_EGOSIGHT_SOURCE_SCREEN_20261028.md)).

**Mandatory design correction — R2-M modality-native and sensor-inclusive fusion experiments (2026-10-12).** This is the next architecture/evaluation workstream, superseding the earlier text-question-first assumption. See [R2-M experiment protocol](docs/pretrained_reuse/R2_MODALITY_NATIVE_FUSION_PROTOCOL_20261012.md). The independently scoped <=10M audio-only experiment was **withdrawn by the user on 2026-10-12**, not put on hold. The archived research proposal is retained solely for provenance, and its training, deployment and integration work are no longer active. The main <=200M Omni model still includes native audio as a required modality. All preserved legacy experiments remain valid in their original scope.

- [ ] **M-0/M-1:** Build direct text-free audio-only, video-only, image-only, screen-only and sensor-only decision probes (plus text-only control) with typed action/option IDs, time/unit/missingness metadata, and no hidden text question or mandatory text encoder. Audio, video, image and sensor each have narrow direct probes. Screen has only oracle-region category pilots that lost to a frequency prior; full screenshot-to-action selection is open. Typed criterion-to-option text control coverage is also incomplete. Treat Audio+Text, Video+Text and sensor-embedding-only results as *non-equivalent* controls.
  Direct text-free CLEVR-4 image taxonomy readout reached 88.30% macro Accuracy on an internal split; count was 58.19%. A separate CLEVRER video-only temporal probe and HARTH sensor comparison are also recorded. These are synthetic/narrow development tasks, not blind product results. The sensor result is imbalanced and transition/class-2 recall is weak. See [image](docs/pretrained_reuse/R2_TEXTFREE_IMAGE_DECISION_20261017.md), [video](docs/pretrained_reuse/R2_TEXTFREE_VIDEO_TEMPORAL_RESULTS_20261018.md), [audio](docs/pretrained_reuse/R2_TEXTFREE_AUDIO_DECISION_20261017.md), [sensor](docs/pretrained_reuse/R2_NATIVE_SENSOR_SUBJECT_RESULTS_20261012.md), and [screen pilots](docs/pretrained_reuse/R2M_SCREEN_CLASSIFIER_RESULTS_20261012.md).
  The new typed-option `NativeFeatureDecisionScorer` was exercised on the fixed eight-taxon iNaturalist image task: 75.35% Accuracy and 0.7253 macro Accuracy, versus 74.88% and 0.7190 for the prior affine scorer. On the same previously exposed development IDs, paired observer bootstrap intervals include zero for both Accuracy measures, though NLL/Brier/ECE improve. This is a post-selection diagnostic and does not close M1; no further model selection should use those IDs. See [diagnostic and hashes](docs/pretrained_reuse/R2_NATIVE_FEATURE_DECISION_DIAGNOSTIC_20261031.md).
- [ ] **M-2:** Test synchronized audio+video, video/screen+sensor, audio+sensor and audio+video+sensor decisions using a gated/time-aware fusion over frozen features. Before fitting, measure option-/metadata-only predictability and confirm the target requires the proposed input streams. Include matched best-unimodal baselines, missing/shuffled/stale/timing-offset/contradictory-signal controls, and a predeclared task **requiring multiple inputs**. STARSS23 is an already measured negative audio/video fusion result, not proof this new matrix is complete.
- [ ] **M-3–M-6:** Add optional text/spoken conditions without making them compulsory, then evaluate individual and combined task quality, <=200M loaded parameters **including the sensor path**, and actual post-quantization >=95% recovery on the same locked task battery. Do not claim general 95–98% world understanding from unrelated narrow benchmarks; lock evaluation before training/selection.


Source follow-up: STARSS23's official MIT release supplies a synchronized
environmental audio/video candidate. The bounded source/decoding probe and
remaining task/split gates are in `docs/pretrained_reuse/R2_STARSS23_SOURCE_20261011.md`.
This does not close joint Decision quality or change existing Teacher corpora.
The subsequent frozen room split covers 49 train / 8 calibration / 21 validation
recordings. Train has no bell positives and validation has only two rooms;
these limits remain explicit in `docs/pretrained_reuse/R2_STARSS23_SPLITS_20261011.md`.

- [x] Cache deterministic frozen text/image/audio features separately for train/validation; preserve sample IDs, source/media hashes and run artifacts outside Git. The typed immutable observation-cache core is exercised on 1,920 audio observations, 1,536 image observations, eight video features, and the state/query/candidate segments of the synthetic text probe; separate CPU tests cover content-key invalidation and payload integrity. Video cache-vs-fresh feature parity now covers eight scenes (16 questions) with zero feature delta on one backend; byte-identical repeat invocation was also confirmed for one scene. Cross-backend video parity and repeated fresh-encoder parity for audio/image remain open.
- [x] Compare final-layer masked-mean, middle-layer masked-mean, and final-layer first-token text readouts with the same variable-option head on identical train/validation IDs (`docs/pretrained_reuse/R2_PATH_A_PROBES.md`). The common `FrozenFeatureCandidateScorer` now lives in `src/tiny_omni_decision/decision.py` and is used by the Path A probe runners; the saved audio multi-query checkpoint reloads through this package head with exact prediction parity. This is one synthetic English task and one seed; learned attention pooling, image/audio pooling, and per-modality readouts remain untested.
- [x] Test many questions on one cached visual/audio event; count avoided encoder executions, memory residency and latency, not merely Accuracy. CLEVRER has 88 descriptive questions over 8 reused validation scenes (11 per cached scene), with 80 hypothetical encoder executions avoided and parity on the original 16 questions. The typed persistent cache roundtrip used those eight features; direct cached-vs-fresh parity now covers 8 scenes / 16 questions. A CPU-only run routes all 88 queries through the byte-bounded LRU into the real frozen candidate scorer: 80 LRU hits, 8 persistent reads, 4 evictions, and bitwise-identical outputs. Audio multi-query uses 640 fixed validation clips and four question types per event (2,560 queries); its packaged frozen runtime now passes parity at batch size 8 with 9.91 ms per observation and 380 MB peak inference allocation. The bundle payload is 227.1 MB, above the aspirational package target. CLEVR-4's 512 validation images each reuse one persisted visual observation across four taxonomy questions; all 2,048 validation predictions match after cache reload. Compact full multimodal packaging and full image/video encoder-plus-readout residency remain open. See `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Measure whether the fixed Audio+Text multi-query readout depends on its cached audio observation using same-ID zero-audio and other-clip-audio controls. On all 640 validation clips and four queries per clip, original macro Accuracy/NLL/Brier/ECE-15 were `0.9785/0.0693/0.0323/0.0143`; zero audio changed them to `0.5250/1.2478/0.5713/0.1769`, and other-clip audio to `0.4961/4.0821/0.9709/0.4835`. A 10,000-resample paired bootstrap by 206 speaker groups excluded zero for all macro differences. This is a post-hoc fixed-readout dependence diagnostic on checkpoint-selection validation, not a blind or broad speech result. Details and artifacts are in `docs/pretrained_reuse/AUDIO_MULTIQUERY_FEATURE_DEPENDENCE_20261011.md`; joint audio-video quality remains untested.
- [x] Compare the unchanged Speech Commands Audio+Text runtime on 1,938 LibriSpeech dev-clean four-choice transcript queries with random and CMUdict-phoneme/lexical/length-matched distractors. Accuracy was `0.2652` vs `0.2637`; paired speaker-bootstrap intervals include zero for Accuracy/NLL/Brier/ECE. Both are out-of-distribution transfer conditions, not broad speech scores; the finding does not show an accuracy penalty from hard negatives because the model is near chance in both. See `docs/pretrained_reuse/LIBRISPEECH_HARD_NEGATIVE_TRANSFER_20261011.md`; joint audio-video and package-size gates remain open.
- [x] Assemble task-scoped Audio, CLEVR-4 Image, synthetic Text, and CLEVRER video-feature Decision components. The v4 package records `170,499,716` unique parameters and `687,254,772` bytes excluding the root manifest; the provisional 200 MB target is not met. Audio/Image/Text passed relocated 8-observation parity checks in v3; the v4 Video scorer replayed all 3,014 frozen validation choices over 200 scenes with zero prediction mismatches and max logit delta `4.77e-7`. The Video component consumes cached, pooled 8-frame VideoPrism features; raw-video extraction remains a separate runtime profile. This is packaging parity, not new quality evidence or a unified dispatcher. See `docs/pretrained_reuse/R2_PATH_A_MODULAR_RUNTIME_20261011.md`.
- [x] Report initial synthetic text, CLEVR-4 image, CLEVRER video, closed-set Speech Commands keyword audio, and one-time Japanese Ruri zero-shot dev diagnostics by source/task and independent source groups in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`. The Ruri result is near uniform chance and is not a trained Decision model. Joint tasks and old Gemma reference comparisons remain open.
- [x] Increase CLEVR-4 image training coverage from the prior 1,024-image probe to 8,424 unique train images using frozen VideoPrism-B and the same 512-image validation selector. Selected same-ID Accuracy/NLL/Brier/ECE-15 changed from `0.5737/1.18360/0.53216/0.09024` (V-JEPA) to `0.8789/0.28781/0.15422/0.02187` (VideoPrism); a one-time report on 1,595 other validation images was `0.8806/0.29116/0.15361/0.01487`. The encoder and training coverage both changed, so this is not a causal encoder comparison. CLEVR-4 is synthetic; count accuracy is 61.2% on the report subset. The 131.9M resident parameters meet the provisional engineering range, while 528.3MB of weights exceed the provisional package target. This does not establish natural-image or temporal-video quality; full record is in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Measure whether more unique CLEVRER scenes alone improve the fixed frozen-video readout. A seed-17 256-train/200-validation-scene run completed at 8 frames with frozen encoders; a same-validation-ID comparison against the old 16-scene checkpoint showed improved NLL/Brier but lower Accuracy and worse ECE, while its learning curve showed early overfitting. This is a mixed development result, not a general-video claim or product promotion; source-archive full hashes and V-JEPA checkpoint rights remain open. Full provenance and outputs are in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Run a bounded task probe for the newly audited VideoPrism-B checkpoint on the same small frozen CLEVRER split as the existing V-JEPA probe. Accuracy tied at 0.375 on 16 validation questions; paired intervals were too wide to distinguish the point estimates. See `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Expand the matched VideoPrism/V-JEPA descriptive comparison to the existing 256-train/200-validation-scene split. VideoPrism's Accuracy/NLL/Brier point estimates were slightly better; a 10,000-resample paired scene bootstrap favored its NLL, while Accuracy/Brier/ECE intervals included zero. Because the validation was reused for V-JEPA selection and selected the VideoPrism readout, this is suggestive development evidence only. Both readouts showed overfit at epoch 24. A later fixed-readout comparison on 200 previously unused scenes found near-tied Accuracy and a small VideoPrism Brier advantage; 259/400 question+option strings repeat across old examples despite zero video-scene/media overlap, so it is scene-held-out but not content-held-out or blind. Keep both as research candidates; neither is promoted. Full metrics, hashes, limitations and compute are in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Measure a binary-choice frozen-feature probe for CLEVRER explanatory, predictive and counterfactual questions using the same 256/200 video-scene split, excluding exact normalized train question/candidate duplicates from validation. The selected VideoPrism + MiniLM readout reached 0.519 validation macro Accuracy (95% scene-bootstrap CI 0.495–0.544), with per-task metrics near chance. This is a weak development result on validation scenes previously used for descriptive-readout selection; it does not demonstrate world reasoning or justify shared-Lens/merging work. Full result, limitations and hashes: `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] On a new 200-scene content- and asset-disjoint CLEVRER sample, compare the fixed VideoPrism+MiniLM scorer, a same-training-data text-only readout and a deterministic scene-video shuffle. All macro Accuracy point estimates were about 0.50; scene-shuffling changed NLL/Brier by less than 0.0002 and its paired intervals included zero. This does not establish useful scene-specific video contribution for this readout; see `docs/pretrained_reuse/CLEVRER_VIDEO_TEXT_CONTROLS_20261011.md` for data, metrics, bootstrap and limits.
- [x] Screen CREMA-D as a synchronized audiovisual source. Its acted-affect labels and access/rights review do not make it an approved proxy for general AV reasoning; no media was fetched or added. Keep it review-only and task-mismatched; see `docs/pretrained_reuse/R2_CREMA_D_AV_SOURCE_SCREEN_20261011.md`.
- [x] Reverse the order of the same eight frames with the fixed VideoPrism+MiniLM readout on those same 200 scenes. Forward and reversed macro Accuracy/NLL/Brier were `0.49899/0.69432/0.50117` and `0.49873/0.69433/0.50118`; paired scene intervals for NLL and Brier included zero, and baseline reload reproduced saved probabilities exactly. A read-only comparison of the saved 768D pooled feature caches found mean cosine `0.999759` and mean relative L2 change `0.02013` across the 200 validation assets. This is development-only evidence that pooled features change slightly but the current readout shows little order sensitivity; it is not a backbone capability limit. The subsequent binary-choice parent-question label audit found 0–4 correct statements per parent, confirming that statement-level `[wrong, correct]` metrics should not be collapsed into a single-winner distribution. Further readout tuning on these already-used validations is not warranted; a stronger next quality experiment needs a new task-matched source or task. See `docs/pretrained_reuse/CLEVRER_TEMPORAL_ORDER_CONTROL_20261011.md` and `docs/pretrained_reuse/CLEVRER_BINARY_REASONING_PROBE_20261011.md`.
- [x] Test one predeclared eight-position scalar temporal pooling rule with VideoPrism-B and the fixed MiniLM/scorer on the same 512 train questions and 400-row, 200-scene development holdout. Only eight weights were trained; all four macro metrics' paired scene-bootstrap intervals include zero, with one net extra temporal answer and worse ECE/Brier. This does not justify further tuning on the reused validation set or establish a backbone limit. Report, cache parity correction, hashes, metrics and restrictions: `docs/pretrained_reuse/CLEVRER_TEMPORAL_POOLING_CONTROL_20261011.md`.
- [x] Run the predeclared same-frame reversal control on 200 scenes with one temporal and one static descriptive question per scene, using the fixed expanded readout. Temporal Accuracy changed from `.435` to `.440` (one net item); NLL/Brier changes were near zero and paired scene-bootstrap intervals included zero or touched zero. Forward reload parity was within `1.79e-7`. This remains a reused-validation sensitivity diagnostic; it neither establishes temporal capability nor explains the earlier binary-choice result. Details, hashes, metrics and compute are recorded in `docs/pretrained_reuse/CLEVRER_TEMPORAL_DESCRIPTIVE_ORDER_CONTROL_20261011.md`. The fixed mean-pooled readout shows little evidence of frame-order sensitivity; inspect pooling/task construction before any layer-level attribution.
- [x] Screen synchronized AV sources. YT-NTU-AVQ has 1,620 rated clips, but inspected metadata lacks original source IDs and its labels measure subjective perceptual quality; it remains `review` with no media fetch. UnAV-100 describes over 10K untrimmed clips with time-localized AV events and a YouTube ID/timestamp/split manifest, but underlying video copyrights remain with owners and the project-page CC BY 4.0 claim conflicts with a Hugging Face mirror's CC BY-NC-SA 4.0 card. Qualcomm QIVD describes 2,900 spoken-question video/audio records but only 22 in its Audio-Visual category; its agreement limits use to internal non-profit research, excludes commercial use, and restricts third-party data combination. It remains `review`; no access was accepted and no files were fetched. None of these sources clears the product-aligned joint-AV gate. See the joint-AV screen reports and candidate manifests.

- [x] Complete the fixed VideoSAUR object-slot endpoint probe on Perception Test occlusion decisions:99 train videos/198 questions,50 fixed epochs, then257 development-validation videos/514 questions with no checkpoint selection. Validation Accuracy/NLL/Brier/ECE were `.361868/2.733249/1.000905/.436157`, versus train Accuracy `.757576`. Changed-target pairs had both answers correct on only3/127 validation videos. Normal-minus-mismatched-video paired intervals include zero. Independent recomputation verified all aggregate metrics and saved prediction hashes. This candidate fails the task-quality gate; do not promote it or infer a universal backbone ceiling. The present head aggregates slot-time tokens without explicit cross-time object correspondence; whether representation transfer, correspondence or small-head generalization dominates remains unresolved. Preserve the fixed endpoint and diagnose on training data before any new validation-guided intervention. See `docs/pretrained_reuse/R2_VIDEOSAUR_VALIDATION_RESULTS_20261013.md`.

**Exit:** a reproducible, measured A baseline that meets stated functional requirements; otherwise diagnose missing features/labels before spending on new backbone training.

### R3 — Shared video backbone + modality Lenses B (**only if A has a measurable inefficiency**)

- [ ] Audit exact backbone compatibility for any pretrained Lens; first reproduce the published Lens-with-original-backbone behavior.
- [ ] Try Lens(s) into a frozen V-JEPA-class video backbone with small bounded data, while monitoring negative transfer and temporal probes. **Input-side Lens training needs gradients through a frozen backbone**; set peak activation and GPU-hour gates before launch.
- [ ] Compare B with A under matched tasks and measured costs. If text/ASR quality requires an independent small tower, retain it and count it; do not force shared-backbone purity.
- [ ] Ablate modality tags, timestamps, mid-layer readout and independent modality side channels. Prefer minimal intervention over full-backbone fine-tuning.

**Exit:** B passes H2 and is preferable on the measured Pareto frontier. If not, adopt A and stop sharing work.

### R4 — Selective cross-modal alignment / predictive repair (**only if diagnosed**)

- [ ] Identify whether errors are missing input information, lens–backbone distribution mismatch, timing/synchronization, poor small-head generalization, or genuine missing cross-modal knowledge.
- [ ] If synchronized rights-approved audiovisual examples exist, compare contrastive/event-alignment and MJEPA-inspired latent prediction as **small, bounded auxiliary losses**, preferably without rewriting a massive encoder.
- [ ] Keep unimodal retention tests, held-out audio-only facts and video temporal probes; avoid forcing identical whole embeddings.
- [ ] Only consider JEPA predictor, recurrent world update, LoRA or selective unfreeze when a counterfactual state test demonstrates need and the fixed resource budget allows it.

**Exit:** demonstrable gain on held-out joint decisions, without erasing unique modality information. If no paired data, record R4 as not testable and use A/B rather than fabricate alignment.

### R5 — State reuse and caching (**functionally driven, not assumed**)

- [x] Establish content-based observation caching correctness and multi-query reuse for sequential Image/Video updates. A real-worker probe passed all 14 checks on two images and two scenes: same-path/same-mtime replacement re-encoded, stale hashes were rejected, aliases and restored content reused features, and all 12 decisions matched saved predictions within 2.4e-7. Audio now also caches single-event features across requests: eight repeated four-query requests avoided all eight re-encodes, matched uncached probabilities exactly, and reduced observed median roundtrip from 13.19 to 2.96 ms. Seven Audio update checks and an isolated bundle reload passed. Concurrent streams and timestamped state remain open. See `docs/pretrained_reuse/R5_OBSERVATION_UPDATES_20261011.md` and `docs/pretrained_reuse/R5_AUDIO_REQUEST_REUSE_20261011.md`.
- [ ] If delayed queries need persistence, compare no memory, 16/32/64 compressed latent slots, and a gated update; retain time/location/missingness metadata. No default promise of action-conditioned physics.
- [ ] If a State Predictor helps specified future-state or sensor tests, train the smallest viable module using permitted temporal supervision; report forecasting quality separately from decision accuracy.

**Exit:** a minimal state representation chosen by actual temporal recall and runtime results.

### R6 — Compatible weight integration / structural reduction (**optional**)

**New research option, not a mandate (2026-10-12):** A small learned multi-modal **decision-sufficient bottleneck** may compress tokens/latent state. That **does not** reduce the upstream pretrained Encoder's parameter count if it still runs. To reach a truly smaller model, verify **teacher-free structured width/head/FFN/layer pruning and recovery**, followed by evidence-preserving quantization. The user suggested exploring ~50M after pruning as a stretch, but preserving 95–98% native/fusion quality at such a reduction is untested. See [latent bottleneck / pruning / Samsung sub-1-bit research note](docs/pretrained_reuse/R6_R7_LATENT_BOTTLENECK_SUBBIT_RESEARCH_20261012.md).



- [ ] For any merge pair, verify shape, ordering, architecture, checkpoint genealogy and pre/post-merge source-task functional equivalence. Reject arbitrary averaging across different bases.
- [ ] Where justified, try aligned weights / FS-Merge-like folding with **small validation-safe alignment data**. Count its training and source towers used during merge.
- [ ] Compare merged output to A and B; keep the unmerged model as a reproducible fallback.
- [ ] Measure true resident byte/operation reductions after folding. A theoretical smaller checkpoint is not a deployed model.

**Exit:** H3 and hardware benefit; otherwise document negative result and skip merger.

### R7 — Precision, packaging and runtime (**after quality gate**)

**Quantization and size clarification (2026-10-12):** Plain low-bit quantization changes **bits per existing weight**, not 100–200M original parameters into 20–30M parameters. The ideal weight-code-only envelope (decimal MB) is: for **100M** weights FP16 **200MB**, INT8 **100MB**, 4bit **50MB**, ternary-entropy (~1.585bpw) **19.8MB**; for **200M** weights FP16 **400MB**, INT8 **200MB**, 4bit **100MB**, ideal ternary **39.6MB**. All exclude scales, embeddings/exceptions, activations, kernels and file/runtime overhead. Samsung Research's [LittleBit](https://arxiv.org/abs/2506.13771), [LittleBit-2](https://proceedings.mlr.press/v306/lee26y.html) and [NanoQuant](https://proceedings.mlr.press/v306/chong26a.html) can factorize and binarize **large linear matrices** to achieve sub-1-bit *average bits per original dense weight*, but they have **not** been validated to preserve >=95% Omni Decisions on 100–200M JEPA-family encoders. Small-width matrices may force extremely low ranks. LittleBit official code is **CC BY-NC 4.0**; NanoQuant is **Apache-2.0**. Neither's existing specialized CUDA decoder kernels establish phone/ARM performance. Treat sub-1-bit as a gated, optional research path **after** achieving the uncompressed modality-native and fusion task quality targets. Full formulas and counter-evidence: [research note](docs/pretrained_reuse/R6_R7_LATENT_BOTTLENECK_SUBBIT_RESEARCH_20261012.md).



**Read-only size envelope (2026-10-11):** all three main encoder weight files are FP32. A manifest- and hash-checked storage-only estimate for group-wise INT8 (group size 256; one FP32 scale and one INT8 zero point per group) is ~182.0 MB decimal versus the current 687.3 MB package, leaving ~18.0 MB against the provisional 200 MB target. No weights were quantized and no quality, device, kernel, memory, or latency result was measured; this does not close R7. Details and hashes: [Path A size envelope audit](docs/pretrained_reuse/R7_PATH_A_SIZE_ENVELOPE_20261011.md).

- [ ] Profile FP32/BF16/FP16 reference behavior **per backend**, with supported numeric combinations only.
- [ ] Test 8-bit and 4-bit deployment paths and their *actual kernels* before targeting BitNet-style ternary. Quantized storage without low-bit compute may not reduce runtime.
- [ ] If needed, measure architecture-compatible ternary-QAT plus recovery, preserving high-precision exceptions and optimizer/master state for training. This is **optional**, not a first-stage prerequisite.
- [ ] Count all exception tensors, tokenizers, scales, metadata and adapters in actual package bytes; verify export/reload parity and latency on the target device, not just CPU test fixtures.
- [ ] No unverified claim that a 1.58-bit name implies 1.58-bit whole-model storage or better speed.

**Exit:** measured quality, package, memory and latency on an identified backend/device.

### R8 — Locked release evaluation and handoff

- [ ] Freeze candidate architecture, all tensors, model/processor revisions, thresholds and run ledger **before** unsealing the final audit.
- [ ] Run one final untouched evaluation per frozen release candidate; aggregate by independent scene/source, show calibration, coverage, OOD, mixed-media, missing-media and temporal results.
- [ ] Publish package manifest, all parameter/byte accounting, trade-offs, license obligations, offline functional tests and device-specific benchmarks with reproducible conditions.
- [ ] Declare `PASS`, `FAIL` or `UNCERTAIN` separately for every target task and device. Do not use accuracy from unrelated public benchmarks to claim product task success.

**Exit:** release qualification is supported by a reproducible, unmodified audit record.

## 7. First bounded action queue (before any training or GPU purchase)

1. [x] Create `docs/pretrained_reuse/CANDIDATES.md` listing pinned candidates, rights, parameters and runtime checks.
2. [x] Create `docs/pretrained_reuse/EVAL_PROTOCOL.md` with task sources, group isolation, split rules and initial metrics.
3. [x] Prototype **read-only** inspection of a pinned V-JEPA-B checkpoint, verifying key names, image/video forwards, output tokens, frame behavior and parity (`docs/pretrained_reuse/R1_AUDIT.md`). Checkpoint rights remain unresolved.
4. [ ] Prototype **read-only** ViT-Lens original backbone + released audio Lens with published preprocessing. Its published pair exceeds 12.3GB, fails the current compact deployment/storage gate, and is not needed for frozen Path A; defer it unless B has a measured justification.
5. [x] Start Path A frozen-feature text, image, audio, and a bounded video probe with deterministic group splits. Text pooling variants, visual/audio feature reuse, multi-query scoring, and the four-modality dispatcher have been exercised; the dispatcher now has an eight-distinct-asset runtime distribution. Image/audio pooling variants and persistent state-update behavior remain untested, while no approved product-aligned joint-AV source is available. These gaps keep the full R2 exit open.

The reusable candidate-scoring core is integrated into `src/tiny_omni_decision/decision.py`, and the frozen-feature Path A probe runners call it. A typed immutable observation-cache core is exercised by text, audio, image, and CLEVRER diagnostics. A read-only pack/export API has now been exercised on the 94,636-entry text cache; it does not replace the source cache and is not wired into an end-to-end multimodal pipeline. The byte-bounded in-memory LRU is exercised in a CPU CLEVRER feature-retrieval-to-scorer path with exact output parity and bounded eviction. A hash-verified local v4 assembly contains task-scoped Audio, CLEVR-4 Image, synthetic Text, and a CLEVRER Video feature-input scorer. The Video scorer references, but does not duplicate, the VideoPrism checkpoint already stored for Image; its 3,014-choice validation replay has exact prediction parity to the saved readout within floating point tolerance. A one-video cross-environment smoke also ran the fixed 8-frame decoder/VideoPrism CPU profile and then the scorer on CUDA, exactly matching the cached video feature and saved decision to floating point tolerance. The four-modality dispatcher matched saved outputs on eight distinct validation assets per modality; second Video/Image requests reused the same frozen features. Batch-one latency, memory and parity records are in the [single-example smoke](docs/pretrained_reuse/R2_PATH_A_DISPATCHER_20261011.md) and [eight-asset distribution report](docs/pretrained_reuse/R2_PATH_A_LATENCY_DISTRIBUTION_20261011.md). These are runtime/replay results, not new quality results or service-level tail guarantees. A joint four-modality Path A baseline and LSTM control were trained and evaluated; the baseline scored 30.58% fixed-endpoint macro Accuracy and the LSTM did not improve the predefined macro-NLL selector. The separate GRID AV probe also did not show useful video complementarity. Neither result meets task quality gates or proves general joint reasoning. The 687.3 MB package exceeds the provisional 200 MB target, and runtime profiles remain separate. The state/query text factorization probe is a separate readout configuration and was slightly below the original text baseline on the same validation IDs. Encoders and experiment orchestration remain research prototypes under `scripts/`. Existing `train-decision`, `configs/decision/durable_teacher.yaml` and `configs/recovery/probability_distillation.yaml` remain legacy Gemma-specific functionality. Do not suggest they execute R1–R7 as written.

## 8. Change-control / supersession map

| Existing artifact / branch | New status | Allowed action now |
| --- | --- | --- |
| `README.md` top-level Gemma description | Historically accurate implementation, **not new strategy** | Add pointer to approved roadmap in a separate documentation change |
| `ROADMAP.md` old "ternary first" contract | **Superseded as future product strategy** by this proposal | Review/approve new strategy before merging |
| `docs/DURABLE_TEACHER.md`, `docs/PHASE3.md`, `manifests/teachers/*` | Historical factual evidence | Preserve intact |
| `docs/DATASETS.md`, allowed-source manifests, source leakage checks | Reusable compliance and dataset infrastructure | Reuse; extend for synchronized media only after rights audit |
| `configs/decision/*` and existing Gemma teacher / EmbeddingGemma QAT implementation | **Legacy experimental track** | Keep reproducible, do not delete or silently repurpose |
| Open PRs #9–#14 and `codex/roadmap-compression-video-specialization` | Concurrent, potentially active and/or stacked research | Review one by one against this new direction; **no automatic merge/close/stop** |
| Proposed A/B/C implementations and R1–R8 | **Not built** | Author specs and bounded work packages after roadmap review |

## 9. References and why they belong here

Primary references and official implementations (paper merit does **not** imply immediately reusable weights):

- [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482), [official V-JEPA source and pretrained models](https://github.com/facebookresearch/vjepa2): small distilled temporal vision starting point.
- [MJEPA](https://arxiv.org/abs/2606.25225), [official code](https://github.com/facebookresearch/MJEPA): evidence for shared audio/video learning and cross-modal prediction, **not** proof of a released suitable small weight file.
- [ViT-Lens](https://arxiv.org/abs/2311.16081), [code/model zoo](https://github.com/TencentARC/ViT-Lens): pretrained ViT adaptation via modality Lenses.
- [LAVISH](https://arxiv.org/abs/2212.07983), [code](https://github.com/GenjiB/LAVISH): frozen visual backbone, parameter-efficient audiovisual adaptation.
- [Perceiver IO](https://arxiv.org/abs/2107.14795), [official implementation](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver): optional query/state bottleneck; **not** an off-the-shelf learned world model.
- [VATT](https://arxiv.org/abs/2104.11178), [official code and model links](https://github.com/google-research/google-research/tree/master/vatt): single-backbone multimodal reference; legacy TensorFlow stack.
- [Model Soups](https://arxiv.org/abs/2203.05482), [Git Re-Basin](https://arxiv.org/abs/2209.04836), [FS-Merge](https://arxiv.org/abs/2410.01483): averaging, alignment and merging under **different compatibility assumptions**.
- [SSAM](https://arxiv.org/abs/2603.21584): specialist merge with shared-base assumptions.
- [BitEmbed](https://arxiv.org/abs/2606.25674), [bitnet.cpp](https://github.com/microsoft/BitNet): low-bit representation and deployment after quality is established.
- [Project durable teacher report](docs/DURABLE_TEACHER.md) and [rights/corpus report](docs/DATASETS.md): actual historical measurements and constraints; not newly validated performance.

## 10. Disposition

This roadmap is a **model-agnostic, evidence-first design proposal**. Keep the full previous roadmap in Git history; it is not deleted from the project record. The requested product outcome stays constant—**small, high-quality, local, genuinely multimodal typed decision inference**—while the mandatory Gemma/EmbeddingGemma ternary-first implementation path is no longer the assumed route. Since this proposal was opened, a task-scoped frozen Speech Commands audio runtime bundle has been integrated on this branch and passed existing-validation replay parity; see the R2 runtime record. It does not close joint AV or full multimodal packaging.

**Next implementation action:** do not rerun the existing Path A baseline or tune exposed validation IDs. The R0 device/quality targets are documented; R2 still needs a rights-cleared, source-compatible synchronized AV task with genuinely both-required labels and independent groups before further joint fitting. The recent DROID transfer did not show a video benefit over sensor-only; the PE-Core candidate did not beat Path A on count and has an implementation-license restriction. Continue task-specific evaluation only on an admitted source, with matched unimodal and input-perturbation controls fixed before fitting. The current four-component assembly is about 687.3 MB, above the provisional 200 MB target; any size reduction must preserve task-specific outputs and be measured.


### STARSS23 measured update (2026-10-11)

All 78 approved paired recordings and 2,720 five-second windows were processed
with frozen audio/video encoders. The preregistered readout comparison completed:
audio-only validation NLL 0.2704, video-only 0.4108, joint 0.3290. Joint input
did not improve audio-only; zeroing joint video improved NLL to 0.2978.
This closes the initial paired-data execution gap, but not the joint-quality
gate. Two validation rooms, class imbalance and event-specific losses limit
interpretation. See [STARSS23 comparison](docs/pretrained_reuse/R2_STARSS23_COMPARISON_20261011.md)
for metrics, curves, controls and artifact hashes. No product promotion,
compression, additional seed or sealed evaluation followed.

### Bounded text adaptation result (2026-10-12)

Approved final-block adaptation of a 22.7M Cross-Encoder completed three epochs.
On the same 2,540 development questions, Accuracy rose from zero-shot 45.98%
to 58.11%, with NLL 1.19869 -> 0.93779. Only 1.775M existing parameters were
updated; frozen hashes and best reload checks passed. Calibration worsened
slightly in the last epoch. This is synthetic English text evidence, not an
Omni quality or quantized deployment result. See
[bounded adaptation report](docs/pretrained_reuse/R2_TEXT_CROSSENCODER_ADAPT_20261011.md).

### Conditional text limitation (2026-10-12)

The fixed adapted Cross-Encoder failed the approved64-state numerical condition
probe:35.94% Accuracy, always selecting B, with0/128 answer changes when switching
lowest/highest. No input truncation occurred. The earlier58.11% synthetic corpus
result therefore does not qualify general conditional Decision behavior. See
[condition probe](docs/pretrained_reuse/R2_TEXT_CONDITION_PROBE_20261012.md).
No additional training or product promotion followed.

### Fixed condition-control follow-up (2026-10-12)

The approved2x2 comparison (base/adapted x bare/described candidate) retained
all256 questions and yielded all-B predictions in every condition. Minimal
candidate descriptions do not fix this failure; it is present before adaptation.
This does not establish a general model-capacity ceiling. Further wording search
on this probe is not justified. See
[full fixed comparison](docs/pretrained_reuse/R2_TEXT_CONDITION_FACTORIAL_20261012.md).

## User-confirmed product contract (2026-10-12)

The user authorizes autonomous strategy, implementation and local experiments.
Routine bounded experiments no longer require repeated design confirmation.
Existing restrictions on paid compute, live-process intervention and preservation
of legacy artifacts remain unless explicitly changed.

Hard architecture target: <=200M underlying resident parameters before
quantization, including all modality towers, embeddings, adapters and heads.
Required inputs: text, video, audio and image; sensors optional. Required quality
target: >=95% on each required modality, aiming for >=98%, for a general
question/criteria-conditioned Decision model. Do not replace this with aggregate
Accuracy, narrow fixed-class taxonomy, or a benchmark chosen after observing
results. Freeze independently grouped task/generalization evaluations before
model selection and retain an untouched final audit. No current candidate meets
this contract. Quantization does not reduce the parameter count for this gate.

Recent NLI reference comparison also failed the fixed numerical diagnostic:
32.03% Accuracy,12/128 fully correct opposite-condition pairs, despite70/128
answer changes. See docs/pretrained_reuse/R2_NLI_CONDITION_20261012.md.
Do not spend further iterations optimizing that diagnostic or deploy either
text scorer as a general reasoning solution. Reassess pretrained objective,
conditional architecture, data coverage and full multimodal parameter budget.

### Instruction-reference result (2026-10-12)

Pinned SmolLM2-135M-Instruct loaded with134,515,008 deduplicated parameters,
but the fixed next-label conditional diagnostic returned A for all256 items
(30.47% Accuracy). This candidate is not selected for multimodal integration.
See docs/pretrained_reuse/R1_R2_INSTRUCTION_REFERENCE_20261012.md for exact
scoring, confidence limitations and parameter accounting. More checkpoint
screening alone is not a demonstrated route to the required quality.

### Full-coverage conditional repair (2026-10-12)

The fixed 2,048-update run completed with 16,384 distinct questions and exact
best reload. Best mean-NLL checkpoint was step768. Curriculum Accuracy changed
24.37% at512 to23.14% at2048; original Accuracy57.83% to57.28%. Slight NLL gains
did not establish useful conditional reasoning. No95% claim or product promotion.
See [full curve and uncertainty](docs/pretrained_reuse/R2_CONDITIONAL_FULL_COVERAGE_20261012.md).

### Encoder-coverage comparison (2026-10-12)

Updating all six existing22.7M-model encoder blocks, with embeddings/pooler frozen,
completed the same2,048-update protocol. Curriculum Accuracy80.57% versus23.14%
for final-block-only; original Accuracy60.24% versus57.28%. Best reload and source
hash checks passed. This is synthetic development evidence; count/conditional
selection remain weak and no four-modality95% claim follows. See
[full curve and controlled comparison](docs/pretrained_reuse/R2_ENCODER_COVERAGE_20261012.md).

### Transfer limitation (2026-10-12)

The selected encoder-coverage weights scored33.59% on the unchanged known-failure
256-question diagnostic, with10/128 fully correct opposite-condition pairs.
Thus80.57% on the training generator's validation does not establish general
Decision reasoning. No truncation occurred; no prompt or checkpoint search followed.
See [transfer evidence](docs/pretrained_reuse/R2_ENCODER_TRANSFER_20261012.md).

### Controlled transfer diagnosis (2026-10-12)

Fixed-weight, paired128-state comparisons found79.79% Accuracy in control,
25.88% after an order-preserving numeric-range shift,54.39% after paraphrasing,
and22.95% with all five perturbations combined. Identifier/field renaming had
small effects. This supports range/wording fragility under the tested conditions;
it does not isolate candidate-count effects or establish broad multimodal quality.
See [all arms, task metrics and uncertainty](docs/pretrained_reuse/R2_TRANSFER_CONTROLS_20261012.md).

### Numeric-range coverage result (2026-10-12)

The numeric-only augmentation run completed2,048updates and best reload. On the
predeclared new transfer set, selected-model Accuracy24.27% versus prior30.66%;
the confidence loss improved but useful reasoning did not. Original curriculum
Accuracy fell80.57%→24.51% at the same budget. No promotion or95%claim. See
[fixed-budget range experiment](docs/pretrained_reuse/R2_NUMERIC_RANGE_20261012.md).

### Numeric training-fit diagnosis (2026-10-12)

The selected wide-range model scores27.25% even on1,024consumed training questions;
the narrow-range model scores82.32% on the same latent states with its own training
values. This supports inadequate fit within the measured recipe, not just failed
transfer; it does not identify a universal capacity ceiling. See
[fixed training-fit diagnosis](docs/pretrained_reuse/R2_NUMERIC_TRAINING_FIT_20261012.md).

### Numeric representation audit (2026-10-12)

The wider training corpus changes numeric vocabulary199→583values and introduces
32.79% two-token numerals, versus0% in the original corpus, with no unknown tokens.
This is a measured representation change, not proof of causation. Before increasing
model size, assess reuse of the learned narrow-format checkpoint under a separately
budgeted, validation-selected wide-range run. See
[contextual token audit](docs/pretrained_reuse/R2_NUMERIC_TOKENIZATION_20261012.md).


The learned-initialization numeric-range control completed2,048updates: selected
narrow-curriculum validation82.23%, final expanded-range training fit83.50%,
with worse original-text NLL/calibration and extra inherited compute. No new
transfer/audit or multimodal quality claim. See
[record](docs/pretrained_reuse/R2_RANGE_WARMSTART_20261012.md).


A fixed selected-weight comparison on new synthetic states found range
interpolation improvement24.12%→71.00%, but wording regression56.54%→42.09%
and poor extrapolation calibration. No general-model promotion.
[Results](docs/pretrained_reuse/R2_RANGE_GENERALIZATION_RESULTS_20261012.md).


Lexical-range control completed2,048updates (selected1408); wording transfer
remains unmeasured for that checkpoint. Separately, frozen TinyCLIP scored48.19%
macro on CLEVR-4 vs9.81% with rotated images; count/texture remain weak.
[Training](docs/pretrained_reuse/R2_LEXICAL_RANGE_RESULTS_20261012.md) /
[image grounding](docs/pretrained_reuse/R2_TINYCLIP_CLEVR4_RESULTS_20261012.md).


The fixed lexical-transfer follow-up found only a small change on its novel
minimum/maximum wording (80.71%→81.05%), with numeric-calibration trade-offs.
It does not repair the previously observed smallest/largest failure by evidence.
[Results](docs/pretrained_reuse/R2_LEXICAL_TRANSFER_RESULTS_20261012.md).


A paired factorial text diagnosis confirms substantial comparison-word
sensitivity at fixed state/range/structure. Lexical training helps but leaves
smallest/largest far below minimum/maximum. No general-quality claim.
[Diagnosis](docs/pretrained_reuse/R2_COMPARATIVE_WORD_RESULTS_20261012.md).


Comparison-vocabulary training completed2,048updates (selected1664). On fixed new
states/templates, smallest/largest improves59.96%→82.81%, but domain terms remain
40.04% (expanded range34.08%). General four-modality95% remains unmet.
[Training](docs/pretrained_reuse/R2_COMPARISON_VOCABULARY_RESULTS_20261012.md) /
[transfer](docs/pretrained_reuse/R2_VOCABULARY_TRANSFER_RESULTS_20261012.md).


Frozen native Whisper transcript likelihood reaches99.95% on the same1,938
random/hard4-choice transcript questions; rotated audio falls to24.82%/22.03%.
No training or parameter increase; calibration and general spoken reasoning remain
open. [Native audio evidence](docs/pretrained_reuse/R2_NATIVE_AUDIO_RESULTS_20261012.md).


The frozen DINOv2 image baseline completed:85.06% across four synthetic taxonomies,
versus87.89% for the existing VideoPrism reference on the same512selection images.
Its39.59M image/text/readout stack is smaller, but count remains53.71%; no replacement
or general-image95% claim. [Results](docs/pretrained_reuse/R2_DINOV2_IMAGE_RESULTS_20261012.md).


The matched-capacity4x4spatial readout gives76.42% vs76.03% for its global control,
with worse NLL and no clear aggregate gain; both trail the smaller-head85.06%.
No adoption or backbone-limit claim. [Spatial control](docs/pretrained_reuse/R2_DINOV2_SPATIAL_RESULTS_20261012.md).

The 134.5M decoder completed2,048updates with exact best reload. Original text
validation improves30.91%→49.92%, but fixed new-state conditional transfer is23.44%
versus81.05% for the22.7M BERT reference; candidate-order consistency is weak.
No replacement, no four-modality95% claim.
[Learning and transfer results](docs/pretrained_reuse/R2_SMALL_DECODER_RESULTS_20261012.md).

Fixed candidate-position averaging on the134.5M decoder costs4.25forwards/question
and gives21.29%/24.71% for normal/reversed options; no consistent Accuracy gain.
Do not adopt or automatically extend training.
[Rotation diagnostic](docs/pretrained_reuse/R2_CANDIDATE_ROTATION_RESULTS_20261012.md).

The matched DINO spatial/global heads completed24epochs with exact reproduction
of the original first8epochs. Selected global81.35%, spatial82.62%; the paired
spatial gain remains uncertain and both trail the smaller-head85.06% reference.
Count remains46.88% for spatial. No adoption or automatic budget extension.
[Learning curves](docs/pretrained_reuse/R2_SPATIAL_CURVE_RESULTS_20261012.md).

At essentially equal readout size, bilinear image/candidate interaction improves
the frozen DINO reference85.06%→87.40% (paired512-image interval+1.03…+3.66pp),
with lower NLL/Brier. Count improves53.71%→59.77%; general image95% remains unmet.
Retain as a measured candidate, without product promotion or video-quality claims.
[Interaction comparison](docs/pretrained_reuse/R2_BILINEAR_RESULTS_20261012.md).

The image-successful bilinear readout did not clearly transfer to pooled video:
macro Accuracy52.24% vs concat52.02% and no-video51.93%, with differences uncertain.
Validation NLL degrades after selected epoch12 despite better train fit. Do not adopt
or extend pooled training. [Video controls](docs/pretrained_reuse/R2_VIDEO_INTERACTION_RESULTS_20261012.md).

The pinned VideoPrism unpooled-token audit passed on two training scenes:
FP32[8,16,16,768], exact reload/repeat and zero global-mean difference from the
old cache. Full456-scene storage would need2.87GB, exceeding current free disk;
no full extraction or quality claim. [Token audit](docs/pretrained_reuse/R2_VIDEO_TOKEN_AUDIT_RESULTS_20261012.md).

RAM transport now passes a full-size allocation check: 456 slots (2.87 GB),
with two real saved tensors transferred exactly and 454 synthetic zero slots.
No live full extraction or quality result yet.
[RAM transport measurements](docs/pretrained_reuse/R2_TOKEN_RAM_TRANSPORT_20261012.md).

The full456-scene token experiment completed three24-epoch comparisons:
attention52.44%, mean52.72%, no-video51.56% selected macro Accuracy.
Paired intervals do not establish an attention/video benefit; do not adopt or
automatically extend training. [Token readout results](docs/pretrained_reuse/R2_TOKEN_READOUT_RESULTS_20261012.md).

Training-only64-example diagnosis reaches100% with attention and97.68% without
video after512updates. This verifies small-subset fit, not generalization or
the product target. [Fit diagnostic](docs/pretrained_reuse/R2_TOKEN_TRAIN_FIT_RESULTS_20261012.md).

Frozen ViLT's full-option8-epoch image probe reaches81.10% versus DINO87.40%;
paired difference−6.30pp. Color improves, texture/count worsen; no replacement.
The curve is still improving, so no capacity-ceiling claim.
[ViLT comparison](docs/pretrained_reuse/R2_VILT_IMAGE_RESULTS_20261012.md).

ViLT24epochs reproduces its original8-epoch prefix exactly and improves to84.47%
(+3.37pp), still below smaller DINO87.40%; ECE worsens slightly. Retain DINO,
with no clean-plateau or general95% claim.
[Fixed learning curve](docs/pretrained_reuse/R2_VILT_CURVE_RESULTS_20261012.md).

DINO/bilinear24epochs reproduces its first8epochs and reaches89.26% (+1.86pp),
with lower NLL/Brier but higher ECE. Exact counting remains64.06% and accounts
for184/220 errors; no tolerance relaxation or general95% claim.
[DINO curve](docs/pretrained_reuse/R2_DINO_BILINEAR_CURVE_RESULTS_20261012.md).

Sampler replay exactly reproduces all1440 updates; accumulation normalization
matches the intended objective. Weight ESS is about37/64 for typical full batches,
but this alone does not establish a cause of poor quality.
[Weight audit](docs/pretrained_reuse/R2_VIDEO_BATCH_WEIGHT_AUDIT_20261012.md).

An alternative pretrained image/text fusion checkpoint, ViLT VQA, passes real
load/forward at117.59M parameters. Closed answers,40-token text positions and
missing temporal/audio capability prevent direct product adoption.
[Runtime audit](docs/pretrained_reuse/R1_VILT_RUNTIME_AUDIT_20261012.md).

ViLT's native answer head lacks9/40 Clevr-4 option slots; none were removed.
Two training images pass frozen question-conditioned feature extraction, enabling
a full-option candidate-head comparison next. Quality remains untested, and
new questions require a joint image/text forward.
[Feature audit](docs/pretrained_reuse/R1_VILT_FEATURE_AUDIT_20261012.md).

Spatial/bilinear reaches90.04% on the reused image selector; matched global88.48%.
Counting improves to70.51%, but superiority over the smaller89.26% reference
is uncertain and ECE worsens. Preserve both; general four-modality95% remains unmet.
[Spatial comparison](docs/pretrained_reuse/R2_SPATIAL_BILINEAR_RESULTS_20261012.md).

Matched physical-mirror augmentation gives90.19% versus90.04% image Accuracy,
an uncertain3/2048-answer gain; NLL worsens and counting drops. No adoption.
[Controlled result](docs/pretrained_reuse/R2_MIRROR_CONTROL_RESULTS_20261012.md).

Shared nonlinear cell pooling reduces head size to180736 parameters but gives
89.79% selected image Accuracy; counting66.41% trails spatial bilinear70.51%.
Later training overfits; no counting-fix or product-quality claim.
[Cell readout result](docs/pretrained_reuse/R2_CELLWISE_RESULTS_20261012.md).

Numeric embedding updates pass a matched512-step run, but both candidate and
control select their unchanged initialization. No quality artifact promoted.
[Numeric comparison](docs/pretrained_reuse/R2_NUMERIC_EMBEDDING_RESULTS_20261012.md).

### Combined candidate budget (2026-10-12)

Separately measured VideoPrism, DINO, Whisper, Cross-Encoder, MiniLM and selected
readout shapes sum to 215,696,513 resident parameters when all retained. This
exceeds the hard 200M limit. A shared Cross-Encoder text-feature hypothesis is
198,306,689 with the same head shapes, but has no combined runtime or quality
verification; existing MiniLM-dependent readouts cannot simply be reused.
See [component audit](docs/pretrained_reuse/R2_COMBINED_CANDIDATE_BUDGET_20261012.md).

### Deterministic numeric-embedding comparison (2026-10-12)

Both fixed2048-update runs completed and selected step1664. Numeric embedding
updates improved the numeric curriculum but worsened original-text selected
Accuracy/NLL/Brier/ECE. Preserve both; no product adoption or95%claim.
See [learning curves, paired diagnostics and hashes](docs/pretrained_reuse/R2_NUMERIC_DETERMINISTIC_RESULTS_20261012.md).

### Shared text/image feature comparison (2026-10-12)

A same-head24-epoch pair completed. Shared Cross-Encoder features yield90.43%
versus90.04% image Accuracy (paired difference uncertain), improved count NLL
but worse color/texture metrics. No product adoption. Sharing is implemented
for the image probe only; video compatibility and the combined <=200M runtime
remain unverified. [Results](docs/pretrained_reuse/R2_SHARED_TEXT_IMAGE_RESULTS_20261012.md).

### Shared text/video comparison (2026-10-12)

All four fixed24-epoch runs completed; prior MiniLM curves reproduced exactly.
Shared text features yield49.79% video macro Accuracy versus50.18% without video.
Neither encoder shows a clear matched-video benefit; later shared training overfits.
No sharing adoption or combined-runtime quality claim. This closes the sharing
probe, not the <=200M / four-modality95% objective.
[Curves, uncertainty and artifacts](docs/pretrained_reuse/R2_SHARED_TEXT_VIDEO_RESULTS_20261012.md).

### Text error attribution (2026-10-12)

A read-only audit of all2540 saved original-text validation decisions finds950
of977 errors in untruncated inputs; score questions reach46.24% exact Accuracy.
Longer context and dismissing errors as label noise lack supporting evidence.
All questions retained; no training or new inference.
[Attribution and limits](docs/pretrained_reuse/R2_TEXT_ERROR_ATTRIBUTION_20261012.md).

### Broad text coverage preflight (2026-10-12)

A real sampler dry-run identifies15127 original questions absent from prior
all-block runs, and produces16384 unique mixed-source encounters. All original
records were used in inherited last-block training. Storage failed:622MB free
versus705MB required, so no learning run started.
[Evidence and retention budget](docs/pretrained_reuse/R2_TEXT_COVERAGE_PREFLIGHT_20261012.md).

### Broader text coverage continuation completed (2026-10-12)

Storage was freed and a separate2048-update run completed, selecting1792 by the
fixed validation rule. Original textAccuracy improves61.54% to63.58%; curriculum
81.54% to84.08%, with lower NLL/Brier/ECE. This is one reused-development-validation
result, not95% or a causal coverage-only claim. Existing multimodal towers remain
unchanged; selected/final artifacts and all curves are preserved.
[Results, uncertainty and accounting](docs/pretrained_reuse/R2_TEXT_COVERAGE_RESULTS_20261012.md).

### Coverage checkpoint fixed transfer (2026-10-12)

A separate frozen10-condition comparison completed on all10240 diagnostic rows
per model. The new checkpoint scores83.98–84.38% on trained-word base conditions,
73.54–74.61% on expanded ranges,40.14% on domain words and36.13% on expanded
domain-word inputs. Every paired Accuracy-difference interval includes zero;
expanded/domain probability quality worsens. No general-reasoning adoption.
[All metrics, opposite-condition pairs and limits](docs/pretrained_reuse/R2_COVERAGE_TRANSFER_RESULTS_20261012.md).

### Question-side failure isolation (2026-10-12)

An actual-file audit corrects the earlier field-name wording: vocabulary family4
never changes state keys. A fixed2x2 diagnostic on8192 rows/model separates the
alias prefix from the remaining domain wording. For the coverage checkpoint,
adding the prefix lowers generic baseAccuracy84.38% to53.03%; domain wording
without the prefix still scores48.83%. All legacy reference probabilities replay
exactly. Both weaknesses coexist; no template-specific fix or product adoption.
[Factorial results](docs/pretrained_reuse/R2_TRANSFER_FACTORIAL_RESULTS_20261012.md)
and [factual correction](docs/pretrained_reuse/R2_TRANSFER_FIELD_AUDIT_20261012.md).


### Compact pretrained NLI check (2026-10-12)

A pinned DeBERTa xsmall checkpoint has **70,830,722 actual parameters**, including
embeddings. On the unchanged 256-question numerical-condition diagnostic it
scored **30.47% Accuracy**, NLL 1.1430, Brier 0.7001 and ECE 0.2041;
opposite conditions changed no answers (0/128 pairs). No training or prompt
search was performed. This candidate is not adopted, and the result is not
a general capacity ceiling. The four-modality >=95% goal remains unmet.
[Load audit](docs/pretrained_reuse/R1_DEBERTA_XSMALL_20261012.md) and
[fixed diagnostic](docs/pretrained_reuse/R2_DEBERTA_NLI_RESULTS_20261012.md).


### Binary versus variable-option scoring

Saved selected-checkpoint predictions show why reduced-choice Accuracy cannot
replace the full task: the 1,536 four-option curriculum questions have 90.89%
full-choice Accuracy but 96.64% gold-versus-distractor pair wins. Pairs are
constructed using known answers, not an available inference policy or a tested
yes/no reformulation. Retain variable options (including binary questions),
full-task quality gates and calibrated probabilities.
[Cardinality analysis](docs/pretrained_reuse/R2_OPTION_CARDINALITY_20261012.md).


### Matched wording continuation result

Two matched1,024-update runs completed. Both select their unchanged starting
checkpoint; wording views improve endpoint NLL/Brier relative to repetition,
but neither improves the fixed validation selector over initialization. Neither
continuation is adopted. This is not a capacity ceiling or a generalization
claim. [Full curves and comparison](docs/pretrained_reuse/R2_PAIRED_WORDING_RESULTS_20261012.md).


### Joint candidate readout: measured, not adopted

Matched1,024-update runs completed. Joint processing reduced timed loop cost
and memory, but selected original/curriculum Accuracy58.31%/40.09% trails
independent63.54%/84.08%. Reversing candidates drops joint original Accuracy
to40.79%; independent predictions remain exact. Reject this candidate, without
claiming a joint-architecture capacity ceiling.
[Curves and fixed order diagnosis](docs/pretrained_reuse/R2_JOINT_LEARNING_RESULTS_20261012.md).


### Training-order intervention (2026-10-12)

A matched 1,024-update option-shuffle run completed with exact best reload.
Original/curriculum Accuracy fell to 54.80%/25.29%; reversing options still
changes 897/2,540 and 1,741/2,048 answer identities. The smaller aggregate order
gap does not establish stability. Candidate rejected; all prior artifacts remain.
[Verified results](docs/pretrained_reuse/R2_JOINT_SHUFFLED_RESULTS_20261012.md).


### Descriptive video grounding (2026-10-12)

A frozen-feature, 24-epoch matched comparison completed on 1,337 training and
2,200 validation questions. Video macro Accuracy38.14% versus no-video35.72%
has an interval including zero; matched versus mismatched video shows a small
+2.32pp scene-information signal. Both overfit and are rejected for product use.
This supports a bounded token-readout diagnostic, not more epochs or a temporal
reasoning claim. [Results](docs/pretrained_reuse/R2_VIDEO_GROUNDING_RESULTS_20261012.md).


### Completed descriptive token comparison; transition to R2-M

All three frozen-token modes completed24epochs. Attention39.99% versus mean38.91%
has an uncertain difference; both exceed no-video35.83% on development metrics.
Attention clearly overfits. Preserve as optional-criterion evidence, do not
extend this family; prioritize the new modality-native/sensor-inclusive R2-M.
[Measured report](docs/pretrained_reuse/R2_GROUNDING_TOKEN_RESULTS_20261012.md).


### Native video/sensor fixed comparison (2026-10-12)

VIDIMU frozen-feature video and joint readouts completed100epochs each on
130train/39validation trials. Selected Accuracy: sensor100%, video66.67%,
joint76.92%; NLL0.164/0.777/0.572. Fixed modality shuffles show both inputs
affect the joint model, but fusion is worse than sensor-only and fails adoption.
This narrow13-class,3-validation-subject result does not establish general Omni
quality or synchronized fusion. No extra seeds or tuning; preserve negative
results and all artifacts. [Full comparison](docs/pretrained_reuse/R2M_VIDIMU_COMPARISON_20261012.md).

### Fixed gated fusion readout (2026-10-12)

A separate preregistered100-epoch run completed on the same frozen VIDIMU split.
The421,826-parameter readout selects epoch100: Accuracy89.74%, NLL1.729,
Brier0.732, ECE0.706. Probability quality is worse than both singletons and
scaled linear fusion; do not adopt. Both losses still improve, so no plateau
claim. [Curves, controls and numerical-check limits](docs/pretrained_reuse/R2M_GATED_FUSION_RESULTS_20261012.md).

### Separate gated learning-budget curve (2026-10-12)

A preregistered fresh1,000-epoch run exactly reproduced all101 earlier prediction
snapshots. Best395 reduces NLL1.729->0.542 but Accuracy89.74%->82.05%; thereafter
train loss falls while validation NLL rises to0.665. This demonstrates budget
sensitivity followed by overfitting, not a model-family ceiling. Sensor-only
still wins; no adoption or further extension.
[Fixed endpoints, selected checkpoint and controls](docs/pretrained_reuse/R2M_GATED_BUDGET_RESULTS_20261012.md).


### Shared spatial image comparison (2026-10-12)

Two fixed 24-epoch readouts completed on the same 8,424 train / 512 selector
images. Shared VideoPrism global/spatial Accuracy is 90.77%/90.92%; the paired
+0.15pp difference is uncertain. Spatial counting improves but texture declines,
with a larger train/validation gap. Keep both artifacts; no product adoption or
95% claim. The image stack is 133,475,584 pre-quantization parameters; a complete
Omni runtime remains unverified. [Curves, hashes and controls](docs/pretrained_reuse/R2_SHARED_SPATIAL_RESULTS_20261012.md).


### Shared global runtime export (2026-10-12)

Algebraically folding repeated global cells reduces the image readout from
1,720,576 to 246,016 parameters without changing any of 2,048 saved answers.
Eight raw images reproduce cached features exactly and produce32 matching typed
JSON decisions. This is a fixed-taxonomy runtime check, not a general Omni
quality result. [Measured export and remaining gates](docs/pretrained_reuse/R2_SHARED_GLOBAL_RUNTIME_20261012.md).


### One shared visual encoder process (2026-10-12)

A single114,365,184-parameter VideoPrism tree now serves explicit image and
video profiles. Alternating raw image/video requests exactly reproduce prior
features, with unchanged weight hashes and a measured1.20GB process peak.
This verifies feature sharing, not joint quality or physical backend buffer
deduplication. [Execution and timing limits](docs/pretrained_reuse/R2_SHARED_VISUAL_ENCODER_20261012.md).


### Fixed full-image preprocessing comparison (2026-10-12)

A single matched24-epoch global readout completed using full-image resizing.
Accuracy90.72% versus cropped90.77%, with uncertain aggregate metric differences.
Counting66.41%->67.58% is also uncertain and texture declines. Preserve the
alternative, retain the existing runtime reference, and stop this preprocessing
comparison without an extra variant or budget extension.
[Curves, paired task intervals and hashes](docs/pretrained_reuse/R2_FULL_IMAGE_RESULTS_20261012.md).


### Training-only counting diagnosis (2026-10-12)

On all8,424 training images the frozen global readout has exact counting68.02%,
with2,001 overestimates and693 underestimates. An affine fixed-candidate rewrite
changes no answers within declared numeric tolerances. No new training or
validation evaluation was performed. Optimize a controlled training-only linear
baseline before attributing the remaining weakness to encoder capacity.
[Confusion matrix and limits](docs/pretrained_reuse/R2_COUNT_TRAINING_DIAGNOSTIC_20261012.md).


### Training-only affine optimization (2026-10-12)

A fixed 200-iteration CPU fit on the same frozen image features improved training
count Accuracy 68.02% -> 78.17% and aggregate training Accuracy 91.43% -> 94.46%.
ECE worsened and the convergence gate was not reached. No validation or audit was
loaded; preserve the runtime reference and do not call this generalization gain.
[Protocol, curves and integrity](docs/pretrained_reuse/R2_AFFINE_FIT_RESULTS_20261012.md).


### Fixed affine endpoint validation (2026-10-12)

The frozen 200-iteration endpoint did not transfer its counting training gain:
reused-validation counting Accuracy 66.41% -> 65.82%, with uncertain paired
Accuracy/NLL differences and a larger train-validation gap. Aggregate NLL improves,
but ECE worsens. Preserve the original runtime; stop unregularized extension and
predefine regularization before further fitting. No audit or promotion.
[Matched metrics, gaps and intervals](docs/pretrained_reuse/R2_AFFINE_ENDPOINT_RESULTS_20261012.md).


### Anchored affine regularization (2026-10-12)

Two fixed 200-iteration fits improve counting on reused validation from the
unregularized 65.82% to 70.51%; paired Accuracy/NLL intervals favor regularization.
Coefficient 0.01 has aggregate Accuracy 91.94%, NLL 0.1964, and is the predefined
research recommendation. This remains far below the weakest-task target and is
not product promotion or a blind result. Stop coefficient refinement on these IDs.
[All arms, curves, gaps and limits](docs/pretrained_reuse/R2_AFFINE_REGULARIZATION_RESULTS_20261012.md).


### Actual shared-visual residency (2026-10-12)

A live two-worker CPU smoke holds the shared image/video backbone, text/speech/
event-audio encoders and conservative head tensors at 195,420,792 elements,
within the pre-quantization 200M budget for those components. RSS sum is 1.81 GB
at the snapshot. A replacement checkpoint and Windows launcher PID initially
caused double counting and memory undercounting; the corrected run preserves both
attempts. This is capacity evidence, not a wired or quality-qualified Omni model.
[Exact ledger and integration gaps](docs/pretrained_reuse/R2_SHARED_RESIDENT_BUDGET_20261012.md).


### Shared raw visual decisions (2026-10-12)

One visual worker now handles raw images and a raw video with the existing heads:
32 image decisions and8 video decisions match saved outputs, with zero answer
changes; repeated images are exact. The video path explicitly retains2048 tokens
required by its attention head. This132.21M functional slice excludes the other
modalities and is a runtime parity result, not new quality or joint-fusion proof.
[Execution, JSON outputs and remaining gates](docs/pretrained_reuse/R2_SHARED_VISUAL_DECISIONS_20261012.md).


### Reusable native speech JSON (2026-10-12)

The native speech runtime now encodes each audio once and shares transcript
scoring across candidate sets, returning custom-ID JSON. Eight existing assets /
16 decisions match the original CUDA reference without answer changes. CPU
probabilities also match but its raw-score gate fails; that limitation is retained.
This is transcript-choice runtime parity, not general audio or joint-Omni quality.
[Both backend results and failure records](docs/pretrained_reuse/R2_NATIVE_SPEECH_RUNTIME_20261012.md).

### Native sensor time-aware JSON (2026-10-12)

The selected static sensor head reproduces eight saved validation outputs with
zero changed answers. The API preserves all twelve classes and explicitly marks
the central label's 1.26 seconds of future evidence; it rejects premature queries.
This verifies delayed-window runtime semantics, not instantaneous or joint quality.
[Replay and limits](docs/pretrained_reuse/R2_NATIVE_SENSOR_RUNTIME_20261012.md).

### Concurrent functional native paths (2026-10-12)

One retained visual worker and its host now execute image/video/speech/sensor
inputs at 169,973,492 resident parameters, preserving saved answers. MiniLM serves
video question/options; standalone text, event audio, screen and joint fusion
are outside this slice. The next gate is task-matched joint quality, not another
residency-only claim. [Measured scope](docs/pretrained_reuse/R2_RESIDENT_NATIVE_PATHS_20261012.md).

### Fusion complementarity diagnosis (2026-10-12)

Saved VIDIMU predictions show zero video-only correct episodes and nine sensor
correct answers lost by the original linear joint model, including two episodes
correct in both unimodal arms. Joint NLL worsens in all three subjects. This set
has no sensor Accuracy headroom; stop fitting additional fusion variants on it
and require a complementary, task-matched paired benchmark before new training.
[Diagnostic scope and next-task gates](docs/pretrained_reuse/R2_FUSION_COMPLEMENTARITY_20261012.md).

### New paired-task source feasibility (2026-10-12)

Official Perception Test train annotations were acquired and hashed: 2,184 videos,
25,757 sound intervals and 17,558 intervals with resolvable source objects.
Source visibility offers a concrete candidate joint task, but contributor grouping
and option semantics remain unresolved. Keep REVIEW; no media, validation or test
was acquired. Answer-reveal cuts are required for 135 train videos.
[Source evidence and admission gates](docs/pretrained_reuse/R2_PERCEPTION_TEST_SOURCE_20261012.md).

Perception Test grouping follow-up: the official paper guarantees script ×
participant grouping, not participant-disjoint splits. The inspected public tree
and train metadata do not expose that mapping. Preserve upstream splits for
development; do not create video-random validation or claim household independence.

The train-only visibility proposal adapter retains all 25,757 annotations,
separates supervision from inputs, and marks 1,063 records held with reasons.
Same-class overlapping events expose ambiguity missed by exact-input checks.
The 24,694 candidates are not admitted training data: oracle event boundaries,
media timing, broad-source semantics and evaluation coverage remain explicit gates.

Visibility task rejected as the primary fusion-quality benchmark: a fixed
leave-video-out sound-class-only prior reaches 96.44% on all train annotations
without audio or pixels. Do not train that proposed head or count the result
toward product quality. Preserve the technical media/features and require a
question-only control for the next evidence-dependent task.

The next prospective task is four-candidate audio-to-video recording correspondence.
The protocol and 256 deterministic train routing records are fixed, including
balanced candidate positions, no source-ID features, unimodal/shuffle controls and
explicit recording-shortcut limitations. This remains an alignment diagnostic,
not general decision quality. See [protocol](docs/pretrained_reuse/R2_AV_CORRESPONDENCE_PROTOCOL_20261012.md).

Correspondence follow-up: the fixed200 official-validation recordings are pinned
with zero train recording-ID overlap. An eight-train-recording projection smoke
passes save/reload and candidate-order controls, using73,728 trainable parameters.
Its100% training Accuracy is memorization evidence only; independent development
quality and content-overlap checks remain pending. [Protocol and all controls](docs/pretrained_reuse/R2_AV_CORRESPONDENCE_PROTOCOL_20261012.md).

The fixed256-train/200-valid correspondence comparison is now complete. Selected
epoch1 reaches51% Accuracy versus25% untrained; later training memorizes train
and worsens validation NLL/calibration. Keep the NLL-selected research probe,
close this comparison, and do not claim general Omni quality or justify greater
capacity from it. [Results and uncertainty](docs/pretrained_reuse/R2_AV_CORRESPONDENCE_RESULTS_20261012.md).

The [post-correspondence requirements audit](docs/pretrained_reuse/R2_POST_CORRESPONDENCE_GATE_AUDIT_20261012.md)
separates completed narrow probes/runtime parity from unmet conditional-semantic,
joint-quality and release gates. Consult it before repeating coverage, transfer,
correspondence or residency experiments.

Dense temporal occlusion comparison completed on 99 train / 257 validation videos.
Mean and attention reach 39.11% / 38.91% selected Accuracy, near the train-only prior;
mismatched videos do not reduce Accuracy and later epochs overfit. Neither head
is promoted. [Results and controls](docs/pretrained_reuse/R2_OCCLUSION_RESULTS_20261013.md).


Frozen TAPIR trajectory plus initial-appearance comparison also completed:
356 clips, seed17,50 epochs, selected epoch11 Accuracy38.91%. Changed-target
pairs remain0/127 jointly correct; later epochs overfit. Train-only ablations
show both feature blocks affect fitting, but do not establish useful identity
reasoning. No promotion or compression. [Results and uncertainty](docs/pretrained_reuse/R2_TRAJECTORY_DECISION_RESULTS_20261013.md).

### Frozen slot and geometry decisions (2026-10-13)

VideoSAUR feature extraction and fixed Decision evaluation completed. Correct
video identity did not demonstrate a resolved quality benefit. On the separate
79-fit/20-observed-development split, an interaction block improved train fit
but lowered development Accuracy from42.5% to30.0%; no adoption.
[Evidence controls](docs/pretrained_reuse/R2_SLOT_INTERNAL_EVIDENCE_CONTROLS_20261013.md).

Slots retain linearly accessible information about their own mask centroids,
including beyond a fixed slot-index prior. Actual decoder replay verifies the
feature/mask correspondence on18 matched frames. This is not ground-truth tracking.
An explicit geometry append was then tested with identical134-input initialization
and fixed50epoch budget: both arms45% development Accuracy; NLL worsens while Brier
improves. Stop this append comparison; preserve both artifacts, no promotion.
[Matched experiment](docs/pretrained_reuse/R2_SLOT_GEOMETRY_COMPARISON_20261013.md).

### Instruction reference follow-up (2026-10-13)

Pinned FLAN-T5-small now loads with all checkpoint tensors verified:76,961,152
parameters, input embeddings shared and output matrix distinct. This local adapter
resolves the installed library's incompatible output tying without modifying the
environment. It does not fit beside current VideoPrism and Whisper under200M.
[Runtime audit](docs/pretrained_reuse/R1_FLAN_T5_RUNTIME_20261013.md).
The fixed256-question diagnostic gives30.47% Accuracy, all answersA, zero correctly
answered opposite-condition pairs. No adoption or additional fit is justified by
that reference result. [Saved metrics](docs/pretrained_reuse/R2_FLAN_FIXED_CONDITION_20261013.md).

Keep the independently scored BERT candidate as the stronger measured text
reference; its lexical/range transfer failures remain explicit. Do not repeat the
completed2,048-update small-decoder experiment or start prompt/seed searches to
rescue these observed diagnostics. R2 still needs useful conditional-semantic and
synchronized multimodal quality; R3/R4 adoption and R7/R8 release remain gated.
Future experiments must specify a distinct intervention and prospective task
contract before fitting, retain evidence-shuffle and candidate-order controls,
and count the complete resident system. None of the new results qualifies the
four-modality95–98% objective or changes the <=200M pre-quantization constraint.

### Text context adaptation and retention (2026-10-13)

All-encoder context adaptation improved the fixed synthetic context diagnostic,
but original-task retention degraded. A fixed384-update, 50% original-task replay
comparison completed and did not resolve the trade-off: original NLL .8356→1.1244,
ECE .0189→.1594. Accuracy difference remains uncertain under paired-state bootstrap.
Preserve the coverage parent; no promotion or additional budget sweep.
[Replay results](docs/pretrained_reuse/R2_CONTEXT_REPLAY_RESULTS_20261013.md) and
[saved-prediction diagnosis](docs/pretrained_reuse/R2_CONTEXT_REPLAY_RETENTION_DIAGNOSTIC_20261013.md).

Fixed soft-teacher replay subsequently completed384updates with identical initial
weights, ordered examples and evaluation. Original NLL improves1.1244→.8598 and
ECE .1594→.0303 versus hard replay, but remains worse than the parent .8356/.0189.
No promotion; retain all comparisons. [Result](docs/pretrained_reuse/R2_SOFT_REPLAY_RESULTS_20261013.md).

Frozen transfer follow-up closes the context-adaptation comparison: familiar-prefix
Accuracy improves, but domain wording remains poor and numeric extrapolation is
only43.3% with NLL3.03. No further budget extension or product promotion from these
in-template gains. [Vocabulary](docs/pretrained_reuse/R2_SOFT_REPLAY_PREFIX_TRANSFER_20261013.md)
and [numeric range](docs/pretrained_reuse/R2_SOFT_REPLAY_RANGE_20261013.md) results.

Screen context follow-up: same-capacity crop+screen classification failed versus
crop+zero. Independent replay verifies final train98.9% versus validation23.0%
and NLL .092/6.378, confirming overfitting in this pilot. Close capacity/budget
extension on these screens; task/representation and app diversity remain open.
[Fixed-set gap and reload](docs/pretrained_reuse/R2_SCREEN_CONTEXT_GAP_20261013.md).

Screen source expansion improves pooled metrics but remains app-dependent and
below a training-frequency prior. Audited TinyCLIP zero-shot also fails class
coverage. Both models miss305/723regions; even label-informed top1 routing reaches
only57.8%, not a deployable quality result. Close these screen readout/reference
comparisons; no merging/residency expansion or prompt tuning is justified.
[Complementarity and limits](docs/pretrained_reuse/R2_SCREEN_COMPLEMENTARITY_20261013.md).

### Native AV evidence-dependent correspondence (2026-10-15)

The prior joint command readout preserved every answer with a constant training
visual state. A controlled native correspondence task now uses an audio query
and four same-speaker/condition videos, all four balanced query roles per quartet.
Frozen audio24/video_ordered18,122790899 AV parameters/coefficients, no new training:
random94.85%/NLL.1450 versus near-command84.08%/NLL.4133. JSON option order is exact;
content swaps under fixed IDs break original answers and corrected labels restore
matched metrics. Missing-input25% is an explicit uniform policy, not trained
unimodal evidence. Nine observed development speakers and repeated quartet roles
limit inference; hard-task train/dev gap10.18pp remains. Primary/replay complete,
with serialization failure and its separate correction retained. Keep as an AV
baseline; independent task/OOD coverage and the complete four-modality product
remain open. No checkpoint/temperature/negative tuning from these scores.
[Actual results and prospective protocol](docs/pretrained_reuse/R2_LOMBARD_CORRESPONDENCE_RESULTS_20261015.md).

### Actual native AV input and state reuse (2026-10-15)

The frozen correspondence primitive now accepts actual archive audio/front-video
bytes through CPU Whisper encoder and an isolated persistent native VideoPrism
worker. Six TRAIN assets /eight fixed query roles complete, with exact full raw
feature/state/JSON fresh replay and both-process profile/dependency checks.48raw
cache hits and128 further decisions invoke neither encoder. Video features match
the original bank exactly; video logits match same-head B1 recomputation exactly
and differ from original stored temporal logits by up to1.91e-6. Audio CPU/CUDA
differences remain explicit.122790899 AV parameters/coefficients include both full
heads; score-only latency is not end-to-end native latency. No new training,
independent quality, full four-modality/package or release qualification. Retain
this runnable reference while independent task/OOD quality coverage stays open.
[Measured runtime, resources, correction and hashes](docs/pretrained_reuse/R5_NATIVE_AV_CORRESPONDENCE_RESULTS_20261015.md).

### Fixed TRAIN-only AV reliability coefficients (2026-10-15)

Six nonnegative slot coefficients, original encoders/heads/data/catalogs unchanged,
complete512 fixed full-batch iterations and an exact independent process refit.
Hard-candidate NLL improves.4133→.3983 with Accuracy84.08%→84.11%, but paired
speaker intervals include zero. Random Accuracy94.85%→94.83% and Brier worsen;
combined prospective runtime-eligibility gate fails. Close this coefficient
comparison, retain equal-weight native runtime and every research artifact.
No extra seed/budget/weight search on these development IDs. This does not
establish a representation ceiling; independent joint task/OOD coverage and
complete four-modality95–98% quality remain open.
[Frozen protocol, all curves/controls/strata and result hashes](docs/pretrained_reuse/R2_AV_SLOT_WEIGHTS_RESULTS_20261015.md).

### Conditional candidate budget and independent supervision (2026-10-15)

The declared nine-file semantic/native inventory plus13 calibration scalars
contains195,655,225 stored tensor elements including buffers. This conservatively
fits200M with4,344,775 headroom; full integration, required-task quality and package
size remain open. New condition-relevance data allocates1,024 previously unused
states /28,672 questions, with12-file exact regeneration and unchanged31 old
validation slices plus16 new slices. No native media, fitting or quality result
was added. Calibration covers binding only. Freeze one matched supervision
comparison and native transfer gates next; do not reopen closed prompt/seed
searches. Old selected0 and all prior failure records remain unchanged.
[Budget](docs/pretrained_reuse/R2_CONDITIONAL_CANDIDATE_BUDGET_20261015.md) and
[data result](docs/pretrained_reuse/R2_CONDITION_RELEVANCE_DATA_RESULTS_20261015.md).

### QASC frozen-feature text Decision probe (2026-10-26)

Added a pinned CC-BY-4.0 eight-choice QASC development probe using a frozen
MiniLM encoder and 196,865-parameter readout. Seed-17 validation selected epoch 1
at 21.81% Accuracy / 1.9925 NLL; later epochs overfit (train CE fell while
validation NLL/Brier/ECE worsened). Fact-pair-group bootstrap gives 18.88–24.73%
Accuracy on these exposed validation IDs. This is task-specific text-only evidence,
not general text quality or product evidence. The test shard remains unopened.
The initial runner report miscounted group identities; corrected counts were
recomputed from pinned JSONL and the runner now preserves `source_record_id`.
No architecture promotion follows from this result.
[Protocol, learning curve, split audit, limits and artifact hashes](docs/pretrained_reuse/R2_QASC_TEXT_DECISION_20261026.md).

### Read-only cross-branch R2 reconciliation (2026-10-26)

Reviewed concurrent open source-screen and experiment PRs to avoid repeating
work. CLEVRER plus generated-tone temporal AV reaches83.51% on its narrow,
exposed synthetic development split; this does not qualify natural AV. The
latest KilometerAudio/EvaOmni/AVI-Bench, MAVD/Urbansas, AMI, AVSBench and NIST
ASAPS screens do not currently admit a rights-cleared, adequately grouped
natural-AV development task. CommonsenseQA2 and QASC both show weak narrow
text transfer. Continue source/task admission before another natural-AV fit;
preserve all concurrent PRs and exposed validation IDs.
[Cross-branch evidence and gates](docs/pretrained_reuse/R2_CROSS_BRANCH_EVIDENCE_20261026.md).

**2026-10-29 07:30 JST R2 one-time text-free image readout evaluation:** A
frozen VideoPrism-B image taxonomy head, selected on its own development split
before this one-time evaluation, scored 2,107 CLEVR-4 validation images. Color/texture/count/shape
accuracy was 99.86%/95.59%/59.33%/99.95% (macro 88.68%); count is the weak
head. This split was accessed by other project experiments, so the result is
not blind project-level evidence and will not be used for tuning. It is a
synthetic taxonomy task, not general natural-image or multimodal quality. The
CPU pass used cached features and did no training or checkpoint selection. See
[protocol, results and hashes](docs/pretrained_reuse/R2_TEXTFREE_IMAGE_OFFICIAL_VALIDATION_20261029.md).

**2026-10-29 R2 train-only count-loss comparison:** On the same 842-image
train-derived development split, count-only CE (58.43% / 0.9440 NLL) was nearly
identical to the existing joint four-head CE reference (58.19% / 0.9447). The
matched cumulative-ordinal-NLL arm selected epoch 5 at 34.80% / 1.8402 NLL;
its train Accuracy rose while train/dev NLL worsened through epoch 512. It was
rejected under the fixed NLL selector; official validation and sealed audit
were not loaded. No hyperparameter search follows from this result. See the
[protocol, learning curve, paired intervals and hashes](docs/pretrained_reuse/R2_TEXTFREE_COUNT_ORDINAL_LOSS_RESULTS_20261029.md).

**2026-10-29 R2 Perception Test source-task disposition:** A read-only audit of
previously saved train-source diagnostics found that 647/692 records labeled
`none` lack a parent-object annotation, and replacing audio with same-label
audio from another video changed OOF accuracy by only -0.27 pp (95% video-group
CI -1.79 to +1.27). The existing audio/video source-candidate task therefore
does not establish reliable dependence on both streams or valid negative labels;
do not tune it further or promote its readout. This is train-source OOF evidence,
not an independent task-quality result. See the [artifact-provenance and
disposition report](docs/pretrained_reuse/R2_PERCEPTION_SOURCE_TASK_DISPOSITION_20261029.md).

**2026-10-29 read-only DROID transfer review (open PRs #32/#33):** On 137
later-shard episodes across 14 already-seen sites, frozen VideoPrism+sensor
scored 39.42% Accuracy / 1.7790 NLL versus sensor-only 42.34% / 1.4968. The
paired joint-minus-sensor Accuracy interval includes zero (-8.08 to +0.97 pp),
while NLL and Brier favor sensor-only. This same-site development transfer does
not pass R2-M or justify shared-backbone work; the two PRs remain independent
and unmodified ([PR #32](https://github.com/Unjuno/tiny-omni-decision/pull/32),
[paired site audit PR #33](https://github.com/Unjuno/tiny-omni-decision/pull/33)).
