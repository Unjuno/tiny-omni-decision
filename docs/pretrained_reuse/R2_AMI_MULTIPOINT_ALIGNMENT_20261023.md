# AMI multi-point audio/video timestamp audit — 2026-10-23

## Result

The existing pinned 24-clip AMI pairing sample was checked at three positions
within every headset-audio clip: 10%, 50%, and 90%. At each position, the
nearest decoded presentation timestamp (PTS) was measured for each of the
meeting's four close-up cameras. All media inputs were verified against their
previous receipts before decoding.

| Split | Audio clips | PTS observations | Mean absolute delta | Maximum absolute delta | > 1 nominal frame (1/15 s) |
|---|---:|---:|---:|---:|---:|
| Train | 12 | 144 | 17.69 ms | 40.00 ms | 0 |
| Validation | 12 | 144 | 18.11 ms | 40.00 ms | 0 |
| Total | 24 | 288 | 17.90 ms | 40.00 ms | 0 |

Overall median absolute PTS delta was 17.50 ms (train: 17.99 ms; validation:
16.00 ms).

The maximum observed nearest-frame distance is below one nominal 15-fps frame
period (66.67 ms). This supports timestamp-grid proximity at the sampled
utterance intervals. It does **not** establish waveform-to-camera synchronization
or that the mapped participant is visible or speaking. In particular, it does
not repair the two empty-room target-camera cases found in the prior blind
train-only screen. AMI remains inadmissible as a model-quality benchmark until
the visibility/label gate and independent-group coverage are addressed.

## Frozen inputs and command

The audit was CPU-only and used the unchanged inputs from the prior preflight:

- Dataset revision: `46f28f2503e2ec48f8867a84eef356c70476beab`.
- Audio receipt: `C:\CodexArtifacts\pretrained-reuse-r2\ami-headset-av-preflight-attempt-15\av-pair-receipt.json`, SHA-256 `649e07132e8f7137638a014eb85f7983dc007ddf099824a9fda2453fb5fccdb4`.
- Video manifest: `C:\CodexArtifacts\pretrained-reuse-r2\ami-rm-camera-preflight-attempt-12\media-manifest.json`, SHA-256 `72d6eabf1679cfcbf0e0cff3bfc4e930b341ac6a7b352e381c1cc3aad0bc3061`.
- Video decode receipt: SHA-256 `757d9514ae9d2957cd975c2edc3715f24cb1383e04a93f777f26ad5ec41b41b2`.
- Audio: 24 mono 16 kHz WAV clips; video: 24 verified RealMedia files across six meetings.
- Authoritative output: `C:\CodexArtifacts\pretrained-reuse-r2\ami-av-multipt-alignment-attempt-18\multipt-alignment-receipt.json`, SHA-256 `5814b985e19bdaf2126c9e879c11c394f3cb5a1518a2824c683d9ec783485255`.
- Receipt records 24 audio clip IDs, meeting/split IDs, sample timestamps, and 288 per-camera PTS deltas; it contains no transcripts or model predictions.
- Source worktree commit: `9966302c06ebacae30d292b88d13c98d03a5d021`; runner SHA-256 `fc18cda18d7251a5c0dbecf7a9a14e65ba4e2388232d499ae5800d9ec4535ba2`.
- Runtime: Python 3.11.9, PyAV 18.1.0. Wall time was about 4.2 seconds; GPU/cloud use and model fitting were zero.

Command:

```powershell
py scripts/audit_ami_av_multipt_alignment.py `
  --audio-dir C:\CodexArtifacts\pretrained-reuse-r2\ami-headset-av-preflight-attempt-15 `
  --audio-receipt C:\CodexArtifacts\pretrained-reuse-r2\ami-headset-av-preflight-attempt-15\av-pair-receipt.json `
  --video-dir C:\CodexArtifacts\pretrained-reuse-r2\ami-rm-camera-preflight-attempt-12 `
  --video-manifest C:\CodexArtifacts\pretrained-reuse-r2\ami-rm-camera-preflight-attempt-12\media-manifest.json `
  --video-decode-receipt C:\CodexArtifacts\pretrained-reuse-r2\ami-rm-camera-preflight-attempt-12\decode-integrity-receipt.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\ami-av-multipt-alignment-attempt-18
```

## Limits and next gate

The three points are local samples inside 24 short clips from three train and
three validation meetings. They do not measure long-recording clock drift,
lip-sync, visible-speaker correctness, or generalization to unseen meetings.
The validation frames were only decoded for PTS and dimensions; their visual
content was not inspected. No examples were filtered or relabeled, no model was
run, and no validation metric or sealed audit was accessed.

The remaining AMI gate is label validity and sufficient independent meeting
coverage. A timestamp match alone is not a reason to train on this cohort.

Attempt 17 used the same verified inputs and produced identical individual PTS
observations. It remains preserved, but its even-count median used the upper
middle value. Attempt 18 corrected that summary calculation and is the
authoritative receipt cited above; this did not affect any PTS observation or
the reported mean/maximum.
