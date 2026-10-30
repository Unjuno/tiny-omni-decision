# Greatest Hits frozen encoders: actual forward and exact replay

## What executed

Following the [fixed protocol](R2_GREATEST_HITS_ENCODER_PROBE_20261017.md), the same
three TRAIN event-row0 inputs pass real frozen Whisper/EfficientAT and VideoPrism
forwards. Separate fresh processes reproduce **all21 NPY payloads byte-for-byte**:
audio features/positions12, video encoder inputs/features/positions9. All four
invocations exit successfully. No pending process or scheduled fit is claimed.

Source commit: `92405b9d53f4f383c11f5dae3b2eb1a14c366bca`.
Fixed raw-input receipt SHA:
`97c91689effda1e7a080b81ddd4be7738718168493137a565f821371d493c0f6`.
No labels enter an encoder. No validation, TEST, calibration or sealed-audit
payload is loaded. Optimizer updates0; model quality evaluation0; cloud cost0.

| Component | Actual parameters | Per-event feature shape | Runtime |
|---|---:|---|---|
| Whisper encoder |8,208,384|25×384|Local CUDA FP32, eager |
| EfficientAT |1,961,239|1×384|Local CUDA FP32 |
| Audio mel preprocessing |0|Internal frontend |Same CUDA device |
| VideoPrism |114,365,184|2,048×768|Isolated CPU JAX FP32 |

All complete model-state digests are identical before/after each invocation and
across fresh replays. Input/source/weight pins are rechecked after execution.
Three events produce different features in each encoder, which rejects a constant
output accident; this does not establish semantic correctness or useful fusion.
The language model/readout were not loaded or trained in these invocations.
The overall197,097,949-parameter interaction candidate remains unchanged, but
this probe does not measure its simultaneous residency or end-to-end latency.

## Actual resources

| Invocation | Measured runner seconds | Setup/load seconds | OS peak working set(bytes) | Peak CUDA allocated(bytes) |
|---|---:|---:|---:|---:|
| Audio |11.785603|9.835596|1,387,479,040|200,990,208|
| Audio fresh replay |9.044333|7.833019|1,387,646,976|200,990,208|
| Video |16.741085|6.463159|4,137,639,936|CPU only |
| Video fresh replay |14.123005|4.187357|4,138,393,600|CPU only |

Recorded runner time totals51.694025s, excluding interpreter/import time before
the timer, tests, publication and the independent artifact audit. No process
overlap was used for these measurements. Peak CUDA allocation is approximately
191.68MiB, not total driver/device memory. GPU kernel time/electricity were not
measured. All runs pass their fixed900s/8GiB host/2GiB VRAM/2GiB free-space guards.

Per-event audio path times were0.8760/0.0753/0.0882s initially and
0.3837/0.0464/0.0413s on replay. Video times were6.6256/1.1258/1.1250s initially
and6.2962/1.1317/1.1172s on replay. The first video includes JIT compilation.
Video timing also includes preprocessing and saving the encoder-input tensor;
audio timing includes loading the prepared arrays, processing and device transfer.
These three inputs do not establish full-corpus throughput or product latency.

No dependency was upgraded. Audio: Python3.11.9, Torch/torchaudio2.6.0+cu124,
Transformers5.6.2, NumPy2.4.6, safetensors0.8.0, TF32off, deterministic algorithms.
Video: Python3.11.16, JAX/jaxlib0.6.2, NumPy1.26.4, TensorFlow CPU2.18.1,
PyAV18.1.0, psutil7.2.2. Deprecation warnings from existing audio dependencies
remain in the logs; they were not suppressed by upgrading the working environment.

## Input and temporal interpretation

The new source-specific audio function consumes the independently prepared16kHz
and32kHz windows directly. It does not alter the old16/24kHz frontend or route the
event input through16kHz. No input peak normalization/clipping is introduced.
The encoders retain their original mel/log preprocessing. Whisper retains its
30s padding and returns the first25 physical-lattice tokens; this is not evidence
that those tokens contain information only from their nominal20ms bins.

Video retains the historical bilinear short-side288/center-crop policy and the
same eight actual frames.456×256 becomes513×288, then crops112/113 pixels left/right.
Only56.14% of resized horizontal extent survives. The full-extent policy was not
silently substituted after its earlier failed retention comparison. Contact at
the edge may consequently be lost; no task-quality conclusion follows from this
successful forward.

New saved position arrays use a shared0.5s window clock:

- Video: actual selected PTS minus sampled audio-window start, divided by0.5;
  repeat each time across its16×16 spatial token grid.
- Speech: nominal0,20,...,480ms lattice, divided by0.5.
- Event-audio: one pooled whole-window vector, assigned the window center.

These coordinates preserve the observed input timing basis. They do not prove
physical sound/contact synchronization, local receptive fields or learned temporal
reasoning. Global encoder Attention can mix positions. Existing baseline feature
positions and all historical banks remain unchanged.

## Reproduction and evidence

Artifacts are below `C:/CodexArtifacts/pretrained-reuse-r2/`:

| Directory | Receipt SHA-256 |
|---|---|
| `greatest-hits-encoder-audio-01` | `b0035843449a66c509e662b08f249489ca48ebe8f174c9646514074064fd1985` |
| `greatest-hits-encoder-audio-replay-01` | `c0860e1d1d721b347fa7826c976c66a2d746b43c212a2b6dd0ec2065194e1d6d` |
| `greatest-hits-encoder-video-01` | `790c5bccf76d5fd842777291a1fbf2d5d4ead0136f91dbffe1d4ec7bace97502` |
| `greatest-hits-encoder-video-replay-01` | `bfa357f80f7c84e072133ba9f15bbd5150c9af1e6afe28de37caff2bf480012b` |
| `greatest-hits-encoder-verification-01` audit | `91dcec6f86d1416510ea62150c7ff50beb63bfa0620b29a3c7ba96e1f246ebd4` |

The verification directory contains RED12-failure/GREEN12-pass logs, all four
actual runtime logs, independent audit source/results, payload hashes and full
CPU/manifest verification. The audit rehashes994 protected old model artifacts
and7,310 previously published data/input files with no changes. It verifies all
46 new payload/receipt file hashes,21 exact payload pairs, finite shapes and the
time-coordinate calculations independently of the production helper.

Small public evidence:
[`greatest-hits-encoders-20261017.json`](../../manifests/pretrained_reuse/greatest-hits-encoders-20261017.json).
Full arrays/weights/media remain outside Git. Exact command arrays, start/end
timestamps, source revisions and pins are in each receipt. From the clean pinned
checkout, use fresh output directories:

```text
# Existing system Python + native-speech-deps; CUDA_VISIBLE_DEVICES=0, OMP_NUM_THREADS=4
python -m scripts.probe_greatest_hits_encoders --phase audio --output <NEW_AUDIO>
python -m scripts.probe_greatest_hits_encoders --phase audio --output <NEW_AUDIO_REPLAY> --reference <NEW_AUDIO> --reference-sha256 <AUDIO_RECEIPT_SHA>
# Existing VideoPrism venv; repo/src + repo only; CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=4
python -m scripts.probe_greatest_hits_encoders --phase video --output <NEW_VIDEO>
python -m scripts.probe_greatest_hits_encoders --phase video --output <NEW_VIDEO_REPLAY> --reference <NEW_VIDEO> --reference-sha256 <VIDEO_RECEIPT_SHA>
```

Whisper source revision169d4a4341b33bc18d8881c4b69c2e104e1cc0af,
EfficientAT sourcea425fdce92572e602a1d5634799bd9f1f2efa806,
VideoPrism sourcec6eba0ece3dd49a24ffc44582966997a4a59f1d8;
exact checkpoint/processor hashes are re-used from the pinned model audits.

Ruff PASS, targeted28tests PASS, full CPU **2,242 passed/1skipped in30.63s**,
all25 manifest commands PASS. Optional Torch tests are skipped when Torch is not
installed in ordinary CI; these real encoder runs are outside normal CI. Four
format/lint findings were fixed before committing/executing; no runtime failure
or discarded candidate occurred in this bounded probe. The publication-head CI
receipt is saved separately after push, without substituting an earlier head.

## What remains and the next action

The encoder path now works on these real paired event windows. Training admission
remains NOT_ADMITTED: the earlier fixed tasks have no demonstrated both-required
stratum, physical sound/contact offset is unknown, and cross-date actor/site/near-
duplicate grouping is unresolved. Source labels may be ambiguous at the cropped
view. Successful serialization/forward does not resolve these data questions.

Next perform a fixed TRAIN-only source review covering dates, label meanings and
contact timing; keep ambiguous/missing cases visible. Define which interventions
preserve truth and which need new verified labels. Use the result to admit a
scoped native task or reject this source for the complementary comparison. Do not
expand to hours of feature extraction until that scope is defensible. A scoped
material/action/reaction benchmark must not be labeled a proven both-required
benchmark merely because both streams exist.

After source/task admission, retain the original Attention baseline plus the
exchange/gate2×2, unimodal/query-only controls, fixed sufficiently long curves and
four-modality retention. No gate fit, calibration, quantization or product
promotion happened here.95–98%, joint benefit, temporal/OOD and deployment gates
remain open; the persistent product goal is not complete.
