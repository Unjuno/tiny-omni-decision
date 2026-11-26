# Greatest Hits full-extent encoder results and live complete cache

## Completed work

The fixed 21-event TRAIN cohort passes real frozen FP32 VideoPrism forwards with
the full image extent retained. A fresh process reproduces **all 105 saved
payloads exactly**: prepared input, video tokens, positions, selected PTS and PNG
per event. The new streaming media-to-bank path then reproduces all 21 events
against those features and verifies exact flushed-bank reload. No learning,
checkpoint selection, quality evaluation or product promotion occurred.

| Local run under `C:/CodexArtifacts/pretrained-reuse-r2/` | Status | Wall seconds | Load seconds | Peak process working set bytes |
|---|---|---:|---:|---:|
| `greatest-hits-full-extent-01` | PASS, 21 events | 35.510 | 4.554 | 4,123,754,496 |
| `greatest-hits-full-extent-replay-01` | PASS, 105 exact payload pairs | 35.678 | 4.394 | 4,120,547,328 |
| `greatest-hits-video-cache-smoke-01` | PASS, 21 events / 63 original questions | 43.113 | See receipt | 4,285,804,544 |

These times exclude interpreter imports before each runner's timer and subsequent
publication/tests. Warm preprocessing plus forward is about 1.0–1.2 seconds per
event; the first call includes JIT compilation (about 5.9–6.0 seconds). Three
bounded runs use 114.301 seconds of measured runner wall time in total. GPU use
and cloud cost: zero. Existing local i7-12700H host and CPU VideoPrism venv;
Python 3.11.16, JAX/JAXlib 0.6.2, NumPy 1.26.4, PyAV 18.1.0, TensorFlow CPU
2.18.1, psutil 7.2.2. No environment upgrade.

## Fixed conditions and actual inputs

[Protocol](R2_GREATEST_HITS_FULL_EXTENT_PROTOCOL_20261017.md), implementation
source `bc80ecb29a6d5ef6eb670b8772cd10357cd26dbd`. Keep the same original
21 TRAIN events/half-second windows from the previous source review. Of each
15-frame window, the existing sampler chooses indices `[0,2,4,6,8,10,12,14]`.
Each actual PTS is preserved. No frame repetition, retiming or event substitution.

Existing policy `full-extent-bilinear-centered-black-pad-288-v1` maps
456x256 -> 288x162, with 63 black rows above and below. The raw image width is
retained instead of the historical central 56.14% crop. This shrinks details;
it is not an established accuracy improvement. The canvas padding remains in the
contextual encoder output, and spatial positions refer to that canvas. Time
positions use actual sampled PTS relative to the shared half-second clock.

Prepared contact sheets #3 (table edge), #6 (red chair) and #18 (door handle) were
visually inspected. The peripheral context/contact remains in the prepared view.
This qualitative check does not establish that the encoder recognizes those
contacts. It is not a label audit or a measurement of physical AV synchronization.

Frozen VideoPrism contains **114,365,184 parameters**, uses the same pretrained
weights and source revision `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, and has
identical state digests before/after every bounded run. The complete experimental
Omni gate architecture remains 197,097,949 parameters. No head was loaded here.

## Full common cache: execution started, not complete

The [complete cache protocol](R2_GREATEST_HITS_CACHE_PROTOCOL_20261017.md) preserves
all immutable source questions:

| Split | Questions | Unique events | Parent videos | Date groups |
|---|---:|---:|---:|---:|
| TRAIN | 23,707 | 8,102 | 308 | 21 |
| Development validation | 8,875 | 3,049 | 101 | 5 |
| Total | 32,582 | 11,151 | 409 | 26 |

Multiple questions reuse an event's computation; no question, class or difficult
example is removed. All original 17 material / 2 action / 5 reaction options and
targets remain. Pure binding tests reject conflicting IDs/times, question changes,
duplicates and cross-split date/media ownership. The source's unresolved near-
duplicate, actor/site and both-required-task limitations remain explicit.

The new cache decodes native PCM to reproduce the exact nearest-96k-sample window,
then streams overlapping real video windows without losing shared frames. The
21-event smoke proves the decoder/bank transport matches the prior prepared-input
probe exactly. Validation question metadata is read for full-corpus binding even
in this smoke; **validation media is not read during smoke**. Full mode reads the
frozen development validation media. Sealed audit/calibration and official TEST
media/labels remain closed.

Full execution started at `2026-10-16T18:27:12.964870+00:00`, source
`857edcf6f243fb23cd2d34d812036e8b43205dfb`, in a fresh directory:

- Output: `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-video-cache-01/`.
- Log: `greatest-hits-full-extent-verification-01/cache-full.log`.
- Live unified-exec session: **92865**; Python PID at launch: **17848**.
- First verified live observation: **77 / 11,151 events**, 3 / 409 completed
  parents, 224 bound questions, about 106.6 seconds; actual PID command and CPU
  activity confirmed. This is a historical progress snapshot, not a completion
  result. Read the live process and current receipt for subsequent progress.

The tensor plan requires 70,156,025,856 bytes for video features plus positions,
PTS and headers. Free disk before launch was about 426 GB. Estimated warm forwards
take roughly 3.1–3.7 hours, with decoding, writes and hashing additional. Fixed
guards: 8 hours, 8 GiB process peak working set and 16 GiB disk reserve. There is
no new GPU training or cloud use. No existing data or artifacts are deleted.

Each parent is complete only after flush, exact independent bank reload, coordinate
reload and hashes. A partial file is not a valid cache. Full PASS additionally
requires all expected events/questions/parents, all 21 reference events matched,
unchanged frozen state and input/source pins. There is no resume flag; an
interrupted extraction remains incomplete. Never restart simply because an
observation call times out; inspect this actual session/PID first.

## Commands and reproducibility

Use the existing VideoPrism Python venv, `CUDA_VISIBLE_DEVICES=''`,
`OMP_NUM_THREADS=4`, and this checkout plus `src` on `PYTHONPATH`. Fresh outputs
are mandatory. Exact argument arrays, timestamps and source pins are in receipts.

```text
python -m scripts.probe_greatest_hits_full_extent --output <NEW_PROBE>
python -m scripts.probe_greatest_hits_full_extent --output <NEW_REPLAY> --reference <NEW_PROBE> --reference-sha256 <PROBE_RECEIPT_SHA>
python -m scripts.cache_greatest_hits_video --phase smoke --output <NEW_SMOKE>
python -m scripts.cache_greatest_hits_video --phase full --output <NEW_FULL> --reference <NEW_SMOKE> --reference-sha256 <SMOKE_RECEIPT_SHA>
```

The full run and smoke use identical committed code. The subsequent publication
changes documentation/manifest only; running Python/config files remain fixed.
Do not mistake the publication commit for the recorded execution-source commit.

| Receipt | SHA256 |
|---|---|
| Full-extent probe | `1ad1b239fd8ffcf561f287e63b88f50c4e66025a437e2e8bcd4b19b7fb3a5f2f` |
| Fresh-process replay | `debb229d6d5902be88db0652d649d1910f43c1f72b7c3a7d0e4733b03f56521f` |
| Decoder-to-bank smoke | `d0ee01083f70a98be68b3a71c0a9abd68ac9b4b82759cc089e0232ff32bd38db` |

The [small manifest](../../manifests/pretrained_reuse/greatest-hits-full-extent-20261017.json)
links the independent audit, source revisions, exact hashes and incomplete full-
cache status. The audit rehashes **8,437** previous files unchanged and verifies
**277** new bounded-run files, all 105 replay pairs, real PTS, padding and all
21 decoder-to-bank matches. It deliberately does not hash or certify live partial
banks as a complete cache. Media and large arrays remain outside Git.

Tests: seven new preparation-contract tests and twelve binding/window tests pass
after their expected RED failures. Ruff PASS; full CPU **2,267 passed / 1 skipped
in 35.67 seconds**; all 25 existing manifest commands PASS. Five lint findings
(line wrapping and explicit zip strictness) were fixed before cache execution.
No failed encoder run or discarded geometry candidate occurred. Exact publication-
head CI is recorded separately in `greatest-hits-full-extent-verification-01/publication/ci.json`.

## Remaining work

Finish the verified live video cache, then complete independent 16/32 kHz audio
features and fixed query features. Verify explicit event-clock positions when
transporting features to the Decision readout; the older generic-position store
must not silently replace them. Freeze common sampling, adequate learning budget,
validation selection and retention rules before the Attention / exchange-gate
comparison. Use the same new features in every arm.

No gate quality has been measured. Valid both-required semantic interventions,
actor/site generalization, four-modality retention, calibration and deployment
quality remain necessary. The original 95–98% product goal is active and unmet.
