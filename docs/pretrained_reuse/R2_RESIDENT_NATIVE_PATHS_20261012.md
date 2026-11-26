# Concurrent native decision paths

## Measured result

One VideoPrism worker remains alive while its parent retains MiniLM, image/video
heads, Whisper-tiny and the static sensor head and executes real inputs. The
worker waits for an explicit parent release; it is not restarted between image
and video requests. Existing standalone invocation behavior is unchanged.

The functional resident slice totals **169,973,492 parameters before quantization**:

| Component | Parameters |
|---|---:|
| Shared image/video VideoPrism | 114,365,184 |
| MiniLM for video question/options | 17,389,824 |
| Image affine head | 61,480 |
| Video attention head | 393,984 |
| Whisper-tiny speech model | 37,760,640 |
| Static sensor head | 2,380 |

Buffers are additional memory, not trainable parameters. This is not the older
195.42M inventory with unconnected models/head tensors: the component sets differ.
There is no standalone text decision model, event-audio model, screen pathway or
joint-fusion head in this run. MiniLM is used for the video question/options.

## Fixed replay, not a new quality experiment

- Eight images / 32 decisions, one video / eight decisions, then repeated images:
  preserve the previous raw-input parity gates and exact repeated-image features.
- Eight audio assets / 16 random-and-hard-transcript decisions: exactly equal
  probabilities to the previous CUDA replay, and raw scores within 1e-4.
- Eight sensor windows / twelve classes each: exactly equal probabilities to the
  previous CPU replay. Central-label temporal scope remains explicit.
- Zero answer changes, no training, no audit data, no changes to input weights.

The input episodes are unrelated and are **not** presented as synchronized
multimodal evidence. This verifies simultaneous residency and independent paths;
it does not establish beneficial fusion, general semantics or 95–98% quality.

## Compute and reproducibility

CPU visual/text/heads/sensor, CUDA float32 Whisper on the existing local GPU.
Companion audio/sensor inference and final hash checks took 1.643 seconds,
excluding model loading, visual work and process startup. CUDA peak allocated
memory in that interval was 392,974,848 bytes. Parent peak working set was
2,078,363,648 bytes; visual worker peak was 1,079,468,032 bytes. Peaks occur at
different instants and must not be described as an observed simultaneous peak.
Cloud cost zero. No end-to-end latency target is claimed.

Command (existing SoundFile overlay; no dependency installation):

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/CodexWorktrees/pretrained-reuse-omni/src'
python -m scripts.replay_resident_native_paths --output C:/CodexArtifacts/pretrained-reuse-r2/resident-native-paths-01
```

Output contains requests, all decisions, worker logs, config/input hashes and
memory measurements. Execution used base commit 12bca93 plus the three runner
changes saved verbatim under `executed-source/` with SHA-256 in
`executed-source-hashes.json`. Fresh output directories are mandatory.

Ruff passes; 591 CPU tests pass, one skipped. The actual replay is a separate
local integration check outside normal CI. Previous commit 12bca93's two GitHub
Actions runs completed successfully. This new change's CI is checked after push.

## Next gate

Residency alone is now insufficient as a next milestone. The remaining work is
an explicit unified task contract and task-matched joint quality evidence, with
no silent averaging of unrelated taxonomy probabilities. Existing negative
fusion results and weak counting/video/condition-following results remain valid.
