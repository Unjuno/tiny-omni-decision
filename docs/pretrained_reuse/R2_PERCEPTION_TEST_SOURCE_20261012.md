# Perception Test: train-only task feasibility audit

## Decision

Retain as a concrete next paired-task candidate, **REVIEW**, not admitted to
training. Existing CREMA-D screening already excludes that narrow acted-affect
task; do not repeat its acquisition or substitute emotion recognition for the
requested general decisions.

Perception Test has same-video sound intervals, source object IDs and source
visibility labels. A prospective decision task is matching a heard event to its
visible source, with explicit invisible/unknown semantics. This is a candidate
requiring label/synchronization review, not an implemented benchmark. Known
source IDs are supervision only, never encoder input. Empty/unknown parents must
not silently become negative examples. All valid source candidates must remain;
do not select easy clips or discard invisible events to improve accuracy.

The existing multiple-choice questions cannot simply be called AV questions:
the train skill tags have no explicit audio/sound category. Having a soundtrack
does not demonstrate that both modalities are necessary.

## Pinned official evidence

Repository revision `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.
[Official README at that revision](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)
states Apache-2.0 for software and CC-BY-4.0 for other distributed materials.
Preserve attribution, source hashes and upstream citation when making a manifest.
This rights statement does not resolve the task and grouping gates below.

The official train annotation archive was downloaded without media or validation
and test annotations. Its SHA-256 is
`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
It contains only `all_train.json` (132,417,530 uncompressed bytes).
URLs for storage objects are mutable; the archive/cut-map byte hashes bind this
audit in addition to the Git revision. Do not call a storage URL immutable.

## Actual train-only counts

| Item | Count |
|---|---:|
| Videos | 2,184 |
| Three-option questions | 7,392 |
| Videos with sound annotations | 2,149 |
| Sound segments | 25,757 |
| Segments whose parent IDs all resolve to annotated objects | 17,558 |
| Videos with at least one such segment | 2,048 |
| Visible sound sources | 19,170 |
| Invisible sound sources | 6,587 |
| Videos requiring upstream answer-reveal truncation | 135 |

Visibility and resolvable-parent counts are separate properties. No missing cut
map entries, invalid positive cut frames or nonpositive sound intervals were
found by this bounded check. This is not full annotation or media validation.

## Required next gates

1. Recover contributor/recording-environment grouping. The inspected metadata
   exposes video IDs and technical fields, not contributor IDs. Unique video IDs
   alone do not establish household/actor-disjoint train/validation/audit.
2. Define event/source option semantics and invisible/unknown handling before
   scoring. Verify which labels genuinely need audio plus video rather than a
   single-modality shortcut; include fixed unimodal and shuffled-source controls.
3. Honor upstream cut-frame mappings: train/validation videos can reveal answers
   in their endings. Trim both audio and video consistently before extraction.
4. Verify bounded media acquisition with checksums and enough disk. Official
   whole-train video archive is listed as 26.5 GB; do not download it blindly.
5. Freeze grouped splits and task protocol before validation inspection, with
   at least 200 independent scenes when feasible and grouped uncertainty.

## Reproduction and artifacts

`python -m scripts.audit_perception_train --source C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01 --output <fresh-result.json>`

Local directory contains the pinned README, train annotation archive, train
cut map and result.json with all hashes, tags and sound classes. The audit checks
train identities, three-option targets, cut ranges and source-parent references.
Unit testing rejects non-train input. No model inference, fitting, media download,
validation/test access, user form submission or cloud cost occurred.

## Follow-up: official grouping evidence

The [paper's Appendix 5](https://arxiv.org/html/2305.13786#A5) specifies grouping
by the combination of filming script and participant. Alternate viewpoints of
the same performance remain together. It does not promise participant-disjoint
splits: the same participant could contribute different scripts to different
splits. The appendix also explains that question templates recur across splits.

Inspection of the complete repository tree at the pinned revision found README,
license/contribution files, visualization and baseline notebooks, but no
participant-to-video mapping. The inspected train metadata likewise contains no
participant or script IDs. This is scoped evidence about these inspected files,
not proof that such a mapping cannot be obtained elsewhere.

Therefore preserve official splits for any prospective development experiment;
do not randomly subdivide videos from official train and call them independent
validation. Official grouping is useful provenance, but does not meet a stronger
participant/household-disjoint gate. Do not rename official validation as a sealed
audit. A stricter split needs an authoritative mapping; do not infer identities
from faces/voices or guess groups from adjacent numeric video IDs.

No external message requesting metadata was sent. No training or validation
acquisition was initiated. Next independent work can define the train-only task
adapter and truncation contract, while keeping this generalization limit open.
The paper HTML and pinned repository-tree response are preserved alongside the
source audit. This follow-up resolves what upstream guarantees; it does not
resolve the missing fine-grained grouping information.

## Follow-up: proposed query failed an identity check

The interval-only question, asking whether the sound source is visible, is
underspecified when multiple sounds occupy the same interval. A train-only exact
input collision audit found 25,661 unique (video, start, end) inputs, with five
contradictory input groups involving eleven annotations. No deterministic model
can give both targets for exactly the same input. Do not fit that formulation.

Adding the requested sound class to the query yields 25,743 unique inputs and no
exact contradictory groups. This supports a class-conditioned query contract,
not a claim of solved semantics: overlapping intervals, broad `Other` classes,
multiple sources and visibility changes still require definition. Class is an
explicit user query condition; it must not be described as predicted from audio.
Parent IDs and visibility remain supervision, never inference inputs. Do not
remove the conflicting rows or turn unknown parents into negatives.

Run: `python -m scripts.audit_perception_visibility_identity --source C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip --output <fresh-json>`.
The saved `visibility-identity.json` retains every contradiction and input hash.
One regression test covers class-disambiguated and still-contradictory sources,
and rejects non-train input. No fitting, media, validation or audit access.

## Train proposal adapter: all rows retained

`perception_visibility.visibility_records` now separates the inference request
from supervision and provenance. Request fields are task, video ID, interval,
sound class and two visibility options. Ground-truth visibility, parent object
IDs and annotation IDs are not model input. Event boundaries and sound class
are supplied query information, so this is an **oracle-boundary, class-conditioned
visibility proposal**, not autonomous event detection or the original MCQA task.

Running the adapter over all train annotations produced 25,757 records:

| Status | Not visible | Visible | Total |
|---|---:|---:|---:|
| Candidate | 6,123 | 18,571 | 24,694 |
| Held | 464 | 599 | 1,063 |

All records remain in the output; none were silently dropped. Held reasons:
91 annotations overlap another same-class event with contradictory visibility;
977 exceed a conservative video/audio/cut boundary; five satisfy both reasons.
An exact-input collision check alone therefore missed an important ambiguity.
The cut frame itself is excluded conservatively. Intervals are not clipped and
relabelled, since that could change visibility semantics. Actual decoded media
timing is not yet checked, and broad class/unknown-source semantics remain open.

No training admission or quality claim follows. Candidate/held is a proposed
adapter status, not a performance-based data selection. Any future evaluation
must report coverage and held strata, preserve its predefined population and
not inflate accuracy by excluding these cases. Video-level IDs retain grouping;
the official participant/script limitation above remains.

Command: `python -m scripts.prepare_perception_visibility --source C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-visibility-proposal-01`.
The records SHA-256 is
`f2d19f6c148bd44c14d1f9b6949bf129f0e3fd2df300dedac0257a55cef7ac4a`.
Tests cover inference-field separation, conservative cuts, overlapping and
touching intervals, and train-only rejection. No media or evaluation data opened.

## Subsequent bounded real-media probe

Two train videos were subsequently acquired from the official train ZIP via
HTTP ranges. Selection was the first two IDs ordered by SHA-256 of `17:video_id`,
independent of labels and model scores. Each selected file was capped at 32 MB;
an oversized/missing selection would fail rather than substitute another video.
Generation `1685446268139925`, archive size 28,404,087,974 bytes and If-Match ETag
bound reads; ZIP CRC and per-file SHA-256 were checked. The full archive was not
downloaded. Earlier statements about no media refer to the preceding audits.

| Train ID | MP4 bytes | Decoded video frames | Decoded audio samples | Metadata samples |
|---|---:|---:|---:|---:|
| video_3174 | 14,443,857 | 1,050 | 1,680,384 | 1,681,408 |
| video_7855 | 5,147,694 | 641 | 1,034,240 | 1,034,240 |

Both streams start at PTS zero and have strictly increasing decoded timestamps.
Video counts match metadata. The first clip has 1,024 fewer decoded audio samples
than metadata (21.333 ms at 48 kHz). Do not silently assume sample-count equality
or infer a decoder cause without investigation. End differences are approximately
19.663 ms and 4.793 ms: video ends here are estimates using the last PTS plus
one average-frame interval, not measured per-frame duration or synchronization
proof. These two files do not validate the whole corpus or semantic AV alignment.

Command: `python -m scripts.probe_perception_media --output C:/CodexArtifacts/pretrained-reuse-r2/perception-media-probe-01`.
The fresh directory contains both files, manifest with byte hashes, timing.json
and the exact executed runner. Decode uses the existing isolated PyAV environment.
No inference/training/GPU work, validation/audit access or cloud cost occurred.
The task remains REVIEW. The next extraction step must use actual PTS and audio
availability, preserve unknown regions and test cutoff handling before fitting.

## Exact decoded audio clock follow-up

Both selected MP4s start with audio packets at PTS -2048 (time base 1/48000)
and carry `skip_samples` side data. For video_3174, decoded samples (1,680,384)
equal the container stream duration in sample units; annotation count remains
1,681,408. This does not support describing the difference as missing interior
audio. For video_7855, stream duration is 1,034,256 while decoded samples and
annotation count are 1,034,240; these are distinct quantities, not interchangeable.
The exact upstream annotation-counting convention remains unverified.

`audio_interval.sample_interval` uses rational PTS/time bases and decoded sample
counts, rejects gaps/overlaps or requests beyond actual availability, and rounds
sample supports inward to the half-open requested interval. No padding or implicit
resampling occurs. All seventeen annotated sound intervals in the two fixed clips
have valid plans (3 and 14 respectively). This validates audio coverage only;
video cuts and source visibility are separate checks. No model inference occurred.

Output: `perception-media-probe-01/audio-clock.json`, including packet side data,
sample counts and all interval plans. Execute with the existing isolated Python:

```powershell
$env:PYTHONPATH='C:/CodexWorktrees/pretrained-reuse-omni/src'
C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe -m scripts.audit_perception_audio_clock --output <fresh-json>
```

The first invocation lacked PYTHONPATH and failed at import before processing;
setting the existing source path resolved it without dependency changes. CPU tests
cover inward rounding, continuous blocks, out-of-range refusal and clock gaps.

## PTS-bounded joint feature pilot

All seventeen train intervals now have input-only NPZ windows. Video uses actual
PTS inside the half-open query interval, eight uniformly selected indices, PyAV
RGB reformat to 288x288. Audio uses exact decoded bounds, channel averaging and
SciPy polyphase 48-to-24 kHz conversion. Targets/classes are absent from NPZs;
query and supervision remain separate. This does not validate physical AV sync.

Six intervals contain fewer than eight unique video frames (4–6); the eight-frame
profile repeats selected in-range frames explicitly. These short intervals are
not discarded or claimed to contain eight independent temporal observations.
The remaining eleven have eight unique frames. No frame outside the requested
interval is used by timestamp; exposure duration is not modeled.

Both pinned encoders were loaded and executed twice on every window:

| Encoder | Parameters | Feature size | Windows | Repeat max difference | CPU load/extract time |
|---|---:|---:|---:|---:|---:|
| DyMN | 1,961,239 | 384 | 17 | 0 | 8.021 s |
| VideoPrism | 114,365,184 | 768 pooled | 17 | 0 | 46.668 s |

Every output is finite. Audio inherits the existing 24-to-32 kHz sinc/mel
profile, so the full pilot route is 48->24->32 kHz, not a direct conversion.
The original video report inherited STARSS23's inaccurate word `equirectangular`;
these are ordinary perspective frames. Its numerical operation resizes the
already square windows and divides by 255. The runner description is corrected
for future runs; original report and executed source are preserved. No feature
bytes were recomputed to hide that metadata issue.

Artifacts: `perception-windows-01`, `perception-audio-features-01`,
`perception-video-features-01` under the existing local R2 artifact root. Reports
include source/checkpoint/window/feature hashes, frame indices, rational PTS and
sample bounds. Commands are `scripts.prepare_perception_windows --output <fresh>`
and `scripts.extract_perception_pilot --modality audio|video --output <fresh>`.
Use the existing audio dependency overlay for audio and VideoPrism environment
with `PYTHONPATH=<worktree>/src`, `JAX_PLATFORMS=cpu` for video. No new dependencies.

This is a two-video technical pilot, not fusion quality or training admission.
No readout fitting, validation/test access or cloud/GPU computation occurred.
Next requires a fixed task protocol and representative grouped development data,
including short/ambiguous intervals and explicit coverage accounting.

## Decision: reject visibility as the primary fusion-quality task

A fixed query-only Laplace-smoothed class prior, excluding **all annotations of
the queried video** from its class counts, achieves:

| Population | Accuracy | NLL | Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|
| All 25,757 annotations | 96.4359% | 0.117077 | 0.060095 | 0.000599 |
| Proposed 24,694 candidates | 96.5093% | 0.114004 | 0.058734 | 0.000598 |

No pixels, waveform, timestamp or model features are used. The sound-class query
alone carries a strong target prior. Therefore a visibility model reaching the
product's numerical aspiration on this task would not establish AV reasoning.
Do not train the planned visibility head or report these numbers as model quality.
Do not remove dominant classes or rebalance based on this result to force a gain.
Retain the media/feature pilot as infrastructure evidence and the failed task as
an attempted design. The dataset itself is not rejected for all other tasks.

This diagnostic uses previously inspected official train annotations. It is
leave-video-out, not participant-disjoint validation or a blind benchmark.
Fixed alpha=1; unseen classes return 0.5/0.5. ECE is descriptive and does not
establish calibration under distribution shift. Source target proportions and
all predictions/IDs are retained under `perception-query-prior-01`.

Reproduce with `python -m scripts.audit_perception_query_prior --output <fresh>`.
The unit test verifies whole-video exclusion and unseen-class handling. No model
training or new validation/audit access occurred. The next task must require
evidence beyond supplied question categories and pass a query-only control before
scaling acquisition or training. No 95–98% achievement is claimed.
