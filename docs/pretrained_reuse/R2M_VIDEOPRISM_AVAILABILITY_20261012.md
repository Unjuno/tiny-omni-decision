# VideoPrism future-context probe: actual pinned CPU execution

## Question and fixed probe

Can existing frame-indexed VideoPrism output features be treated as available at
their corresponding frame timestamps? Inspect the pinned encoder and test one
known **training** clip, without optimization or quality evaluation.

Model:`videoprism_public_v1_base`,114,365,184 parameters. Source commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; checkpoint SHA256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
The local upstream `encoders.py` explicitly sets `enable_causal_atten=False`
in the video encoder; other model configurations using causal attention are
not evidence that this loaded model is causal.

Input: previously decoded VIDIMU training S40_A01_T01 original video; SHA256
`b6d8fd31bc9459e3f822c4df20a844232db491d260cd82d5e0585af3df3de0d9`.
Use the existing8-frame endpoint sampling and resize/crop. Three inference
calls: original, unchanged repeat, then invert normalized pixels of frames4..7
(`1-x`), retaining frames0..3 bitwise. No alternatives or input search.

## Observed output dependence

Unchanged repeated inference: max absolute token difference **0**.

| Output time index | Token delta RMS | Max pooled-feature absolute delta |
|---:|---:|---:|
| 0 | 0.111398 | 0.250923 |
| 1 | 0.106687 | 0.290099 |
| 2 | 0.124335 | 0.342003 |
| 3 | 0.133551 | 0.349420 |
| 4 | 0.331051 | 0.424214 |
| 5 | 0.330541 | 0.435198 |
| 6 | 0.337683 | 0.434766 |
| 7 | 0.341004 | 0.536411 |

Output indices0..3 correspond to **unchanged input frames**, yet all change
substantially. This directly refutes a causal-per-frame interpretation for
this encoder/input. It does not measure decision accuracy, quantify typical
future dependence, or imply a model bug. A noncausal clip encoder is legitimate
for decisions made after the clip is available.

## Required runtime interpretation

This clip's selected frame times span0..36,266.667ms. All eight output time
slices used by the current flattened episode representation are available only
after the clip's final needed input and actual processing completion. The first
slice cannot be assigned availability0ms or15,533ms based on its frame index.
The new feature-gather adapter cannot remove future information already mixed
inside the encoder.

For an episode-vector adapter, keep the flattened8x768 vector as one feature
row with its true completion/availability time. For a future streaming model,
encode windows containing only observed past inputs and use window-completion
times, then measure latency and quality on that changed input protocol. Do not
silently relabel existing full-episode features, truncate their output tokens
as a causal substitute, or treat hypothetical timestamps as measured runtime.
No windowed re-extraction or temporal-model training was executed here.

## Reproduction and cost

From repository root with `PYTHONPATH` set to its `src` directory:

```text
C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe -m scripts.probe_videoprism_future_context
```

Source parent6e97b27 plus probe; exact executed script hash is saved in config.
A later edit only wrapped a long version-report line. Outputs:
`C:/CodexArtifacts/pretrained-reuse-r2/videoprism-future-context-probe-01/`.
Config pins inputs/source; result stores timestamps, full per-time delta
statistics, versions/timing; pooled features for all three calls are preserved.
Pooled-feature SHA256:
`6939a2d5909fbc59cd253ac94f91b0f39971826826a117cfe9208f1ea47b0eaa`.

Measured forward times6.100s (includes first compile),1.124s repeat,1.171s
perturbed. These are not input-to-decision latency: loading/preprocessing/JSON
are excluded. Peak process working set4,173,369,344bytes; CPU deviceTFRT_CPU_0;
JAX0.6.2, NumPy1.26.4, TensorFlow2.18.1. No GPU/cloud, training, sealed data,
legacy cache changes or model promotion.
