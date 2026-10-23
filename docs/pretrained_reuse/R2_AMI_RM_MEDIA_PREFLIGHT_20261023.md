# AMI low-bitrate camera media preflight — 2026-10-23

## Decision

The six previously selected, meeting-disjoint train/validation groups now have
their four official low-bitrate RealMedia close-up streams locally available
and hash-verified. All 24 files open with PyAV 18.1.0 and decode sampled frames
at 0%, 50%, and 90% of the reported duration. This clears the narrow media
acquisition and sampled-decode gate only. It does **not** establish that the
intended speaker is visible, that every audio interval has a usable matching
frame, or that the audiovisual speaker-binding task is suitable for quality
evaluation. AMI remains a candidate, not an admitted benchmark.

No model inference, training, quality scoring, new audio acquisition, or
sealed/unseen split access was performed. The unrelated Orbitune GPU process
was left untouched; this work used network I/O and CPU decoding only.

## Scope and source

This is a continuation of the six-meeting, participant-headset-audio to
four-camera-speaker-binding candidate documented in
[`R2_AMI_AV_BINDING_PREFLIGHT_20261022.md`](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-ami-speaker-channel-preflight-20261022/docs/pretrained_reuse/R2_AMI_AV_BINDING_PREFLIGHT_20261022.md).
The split remains by meeting, with three train meetings and three validation
meetings:

| Split | Meetings | Close-up streams |
|---|---|---:|
| Train | TS3009a, ES2010b, IS1004b | 12 |
| Validation | TS3005b, ES2008b, IS1006b | 12 |

Only the official Edinburgh AMI corpus mirror was contacted. For each meeting,
the script checked the official directory listing, constrained links to the
four expected relative camera filenames, checked `Content-Length`, streamed
the response into a new `.partial` file while calculating SHA-256, and renamed
it only after the byte count matched. Existing files were not overwritten.
The 24 media files are outside Git at:

`C:\CodexArtifacts\pretrained-reuse-r2\ami-rm-camera-preflight-attempt-12\`

The exact per-file source URL, size, and SHA-256 are in
`media-manifest.json` (SHA-256
`72d6eabf1679cfcbf0e0cff3bfc4e930b341ac6a7b352e381c1cc3aad0bc3061`). The
manifest itself records the pinned AMI audio metadata revision used by the
prior candidate screen; it does not imply that the video files are versioned
by that Hugging Face commit. The sample-decode and independent file-integrity
receipt is `decode-integrity-receipt.json` (SHA-256
`757d9514ae9d2957cd975c2edc3715f24cb1383e04a93f777f26ad5ec41b41b2`). Both
receipts remain in the external artifact directory; no media bytes or full
per-file manifest are committed.

## Integrity and decode findings

- 24/24 expected streams downloaded; no `.partial` files remain.
- Total media size: 312,537,065 bytes (298.06 MiB).
- All local lengths and SHA-256 values exactly match the download manifest.
- All 24 containers expose one video stream and no audio stream. The audio
  must be acquired separately from the participant headset-channel source.
- Every video stream reports RV40 (`rv40`), 360×284, nominal 15 fps.
- Three frames per stream (72 sampled frames total) decoded at the beginning,
  midpoint, and 90% position. This is a seek/decode smoke, not a full-frame
  scan, PTS continuity test, or synchronization measurement.

| Meeting | Split | Shortest–longest camera duration (s) |
|---|---|---:|
| TS3009a | Train | 1495.841–1501.280 |
| ES2010b | Train | 1749.360–1752.240 |
| IS1004b | Train | 2162.401–2166.801 |
| TS3005b | Validation | 2425.441–2432.361 |
| ES2008b | Validation | 2086.160–2089.680 |
| IS1006b | Validation | 2145.281–2147.120 |

Camera duration differences within a meeting are retained as reported; no
trimming or synchronization correction was applied. In particular, the
official AMI data-problems record and previous metadata audit flag `ES2008b`
video truncation relative to audio. Candidate intervals outside the common
audio/video range must not enter a later audiovisual evaluation. The previous
candidate screen also identifies looser synchronization for TNO (`TS*`) than
for E/I recordings. These source-level notes still require measurement against
the exact frame PTS and headset audio used by any future example.

## Reproduction and environment

Source commit: `0bbb5d17cc9a2b7e5e0d9353e3e70d0c6c1ff17a` (roadmap branch head
used to create this work branch).

Download command:

```powershell
py scripts/preflight_ami_rm_media.py --output-dir "C:\CodexArtifacts\pretrained-reuse-r2\ami-rm-camera-preflight-attempt-12"
```

The script is restricted to these six meetings and four camera filenames.
The file integrity and sampled decode verification used Python 3.11.9 and
PyAV 18.1.0. No model weights were loaded, no CUDA operation was run, no
cloud compute was used, and no paid resource was used.

## Remaining gates

The prior speaker-binding preflight remains controlling for task validity:

1. Acquire the matching headset-channel audio for the frozen candidate rows
   without using transcript text as model input.
2. Validate meeting-specific participant → headset channel → camera mapping
   against the exact selected media.
3. Check each candidate's audio interval against decoded video PTS and the
   common recording range. Do not score TNO as tightly synchronized without
   an explicit measured alignment check.
4. Adjudicate whether the target speaker is actually visible and sufficiently
   identifiable in the correct camera. A mapped camera and timestamp alone
   are not a valid positive label.
5. Add audio-only, video-only, time-shift, camera-shuffle, missing-stream,
   and randomized-choice controls, with uncertainty clustered by meeting and
   participant. Three meetings per split are a low-power feasibility set,
   not a reliable broad quality benchmark.

Until these gates pass, do not claim an AMI score, multimodal benefit, or
general AV decision capability from this source.
