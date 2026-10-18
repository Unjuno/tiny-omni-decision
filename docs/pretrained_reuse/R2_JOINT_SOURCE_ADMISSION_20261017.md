# Joint AV source probe: actual media and label findings

## Status and relation to the gate comparison

The new conditional AV readout has **197,097,949** encoder-inclusive reference
parameters, but its current common corpus has **zero joint AV observations**.
This investigation acquires real paired TRAIN media and checks source labels
before fitting that residual. No model was loaded or trained in this probe.
The Attention baseline, all previous fits and the four gate/exchange arms remain
fixed. No quality gain, calibrated JSON or 95–98% product result is claimed.

**Execution: PASS** for bounded TRAIN acquisition and full CPU stream decode.
**Training admission: NOT_ADMITTED / REVIEW**, pending grouping, label/task and
physical alignment checks. A permissive source license is not a validated task.

## Source screen

| Source | Evidence and decision |
|---|---|
| Greatest Hits | Official author page links CC BY 4.0 and released recordings/labels. Selected for the fixed physical probe below. |
| Action Verb Corpus | Institutional page and Zenodo license field say CC BY 4.0, but the same Zenodo record's version text says CC BY-NC-SA 4.0/non-commercial. REVIEW; no training media or annotations acquired. |
| FAIR-Play | Publisher README says CC BY 4.0 and roughly aligned paired clips. Its random splits can share scenes, and a complementary semantic task is not established. No media acquired. |

Primary sources: [Greatest Hits](https://andrewowens.com/vis/),
[paper](https://arxiv.org/abs/1512.08512),
[AVC institutional page](https://www.ofai.at/resources/avc),
[AVC distribution record](https://zenodo.org/records/5140014),
[FAIR-Play pinned README](https://github.com/facebookresearch/FAIR-Play/blob/2666d6caf2d126ae43a84841e94e0c9ae24b3e2d/README.md).
Public metadata snapshots and hashes are retained in the evidence manifest.

Existing Lombard command fits and VIDIMU comparisons remain closed with their
measured limitations. Their high scores did not establish useful AV complementarity.
STARSS23 event-presence labels alone do not establish visual necessity either.

## Fixed acquisition and actual source layout

Protocol committed **090449a**, before member payloads. Acquisition source:
`f029c7539399e4bc8f927ca78fbce58ec8679ce3`.

Archive URL:
`https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip`.
Size **21,142,601,241 bytes**, strong ETag `"4ec327e19-5cc7fa3687c80"`.
The full archive was not downloaded or SHA256-hashed. Directory-only acquisition
transferred 734,903 bytes; member acquisition transferred 2,456,254 bytes in
23.735 seconds. All member reads passed exact HTTP 206/Content-Range/ETag,
decompressed length and CRC checks; each extracted file has SHA256 evidence.
TLS remained enabled. IPv4 requests had explicit wall/process limits.

The actual directory has 977 parents, each with exactly these six members:
`denoised.mp4`, `denoised.wav`, `denoised_thumb.mp4`, `mic.mp4`, `mic.wav`,
`times.txt`. Official split metadata contains 733 TRAIN and 244 TEST IDs,
without duplicate IDs or exact-ID overlap; their union is the full parent set.
Reading `test.txt` read ID metadata only. No TEST media or label files were opened.

The selected TRAIN parents were the first three by SHA256(`17:` + ID), fixed
before seeing their annotations:

| Parent | Annotation rows | All three labels present | All three missing |
|---|---:|---:|---:|
| 2015-09-29-15-44-54-671 | 40 | 23 | 14 |
| 2015-09-30-20-56-02-304 | 29 | 18 | 8 |
| 2015-03-30-00-26-27 | 29 | 13 | 15 |
| Total | **98** | **54** | **37** |

Seven further rows have partial labels. Raw files have no header: one timestamp
and three categorical tokens per line. Material/action/reaction meanings are
inferred from released values and the paper, which describes those annotations.
The probe found leaf/dirt/rock/tile, hit/scratch and scatter/splash/static/
rigid-motion values. No impact-location coordinates are present in these files.
All raw tokens are retained; `None` remains missing, never a negative label.
Times are finite, nonnegative and strictly increasing in each selected file.

## Grouping warning from actual split IDs

| Conservative ID prefix | TRAIN distinct | TEST distinct | Shared |
|---|---:|---:|---:|
| First six numeric components (timestamp) | 419 | 164 | **55** |
| First three components (date) | 28 | 25 | **25** |

324 TRAIN IDs and 134 TEST IDs belong to the 55 shared timestamp prefixes.
These are measured string-prefix overlaps, **not proof of duplicated media**.
What the suffix, scene, actor and session identities mean remains UNKNOWN.
Exact clip-ID disjointness therefore cannot justify independent-scene claims.
Do not split impacts/windows randomly, or silently use official TEST for model
selection. A development partition must preserve all related assets together.

## Physical media result

Decode source `24e51b61598a283eed55ea3874774b2830575481`; selected first parent's
`_denoised.mp4`, SHA256
`a856f1bf7ccc6ea960fa7b81512624956dbb6d8f1982d403affe22507159d444`.

| Property | Measured value |
|---|---|
| Video | H.264; 456×256; 30000/1001 fps; 841 decoded frames, 840 distinct pixel hashes |
| Video PTS | 0 to 28.028 s; decoded end 28.061367 s; strictly increasing |
| Audio | AAC; mono 96 kHz; 2,625 decoded frames; 2,688,000 samples/channel |
| Audio PTS | 0 to 27.989333 s; decoded end 28.000 s; strictly increasing |
| Audio values | float32, all finite; RMS 0.0312706; peak absolute 1.4588075 |
| Annotation times | All 40 inside both decoded stream extents |
| Decode wall time | 3.343 s, CPU only |

The 61.4 ms decoded-end difference and common start PTS do not measure physical
AV offset. Sample peak exceeds 1; no clipping or normalization was applied.
Future resampling/scaling must be explicit, checked and identical across arms.
Sparse raw frame previews show a drumstick interacting with leaves/soil without
visible annotation text in the inspected frames. They do not certify every frame
or every source video. Physical impact/audio alignment is still **UNKNOWN**.

Environment: existing isolated Python 3.11.16, PyAV 18.1.0, NumPy 1.26.4;
FFmpeg library versions are saved in the manifest. Initial decode source a13bc36
failed before output creation because Pillow was unavailable. The failure is
retained; a separate committed correction used the existing PyAV PNG encoder.
No package was installed/upgraded. Cloud cost **0**, optimizer updates **0**.

## What this permits next

The physical source is promising for material/action/reaction Decision questions,
but raw co-occurrence and annotation count do not establish a both-required task.
The next bounded step is a TRAIN-only annotation/group inventory, with every
missing or unresolved record accounted for and no model scores used to filter it.
Establish conservative parent/session grouping and decode/label alignment before
freezing development splits. The 244 official TEST media/label payloads stay closed.

Then define question/candidate mappings and valid missing/conflict/relevant-swap
labels. Native material/action/reaction tasks may support complementary prediction;
joint benefit must be measured against audio-only, video-only and question-only
controls. Do not relabel arbitrary mismatched pairs as definitely wrong, assume
missing fields are false, or equate matching identity with semantic reasoning.
Only after this data gate, freeze matched Attention and exchange/gate learning
curves, input interventions and four-modality retention criteria. No such run has
been launched or scheduled by this probe.

## Reproduction and evidence

Run acquisition in a clean checkout with stdlib Python (new output required):

```powershell
python -m scripts.probe_greatest_hits_train --directory C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-directory-01/members.json --output NEW_ACQUISITION_DIR
```

Decode offline with the existing PyAV environment and `CUDA_VISIBLE_DEVICES=''`:

```powershell
python -m scripts.inspect_greatest_hits_train --probe NEW_ACQUISITION_DIR --output NEW_DECODE_DIR
```

Exact executed interpreter paths, commands, revisions and per-member hashes are in
[the evidence manifest](../../manifests/pretrained_reuse/greatest-hits-probe-20261017.json).
`scripts/report_joint_source_probe.py` recreates that small manifest from the fixed
local receipts and refuses to overwrite it. Raw source files/media remain outside
Git in `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-*`.

Source tests cover unsafe/corrupt members, exact ranges/version checks, bounded
reads, deterministic official-TRAIN selection, timestamp validation and missing
label preservation. The new JSON is checked by CPU regression assertions; the
existing CLI dataset-manifest validators cover the existing catalogs, not this
distinct physical-probe schema. No current trainer/config/corpus/checkpoint changed.

Publication verification: Ruff PASS; **2,212 CPU tests passed, 1 skipped** in
30.09 s; all25 existing manifest commands PASS. SHA256 rechecks of994 previously
protected native artifacts found0 changes. The user's separate OneDrive checkout
still has its pre-existing two modified documents and untracked resume script.
Local verification logs are under
`C:/CodexArtifacts/pretrained-reuse-r2/joint-source-verification-20261017-01/publication`;
the exact pushed commit's CI receipt is saved there after publication.
