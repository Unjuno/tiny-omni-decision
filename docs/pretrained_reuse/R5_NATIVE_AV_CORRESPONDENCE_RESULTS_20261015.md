# Actual native AV correspondence and reusable states

## What actually ran

6 native audio clips and6 front videos from the first TRAIN anchor’s
frozen random/hard quartets, all four query roles each. Full raw ZIP bytes,
native decoding/frontends/encoders, frozen audio24/video18 heads and custom
four-option JSON completed. No text question/transcript/label as model input.
No training or model changes. All old source/corpora/checkpoints remain intact.

Actual backend calls: audio6, video6.
48 repeated raw requests are cache hits;128additional decisions
invoke neither encoder. Profile/media/modality/time keys and bounded LRU
retain immutable owner-bound51-logit states. This is task-specific decision
state reuse, not a general world-state or temporal dynamics model. Metadata
time ranges are local clip ranges, not a newly proved common acquisition clock.

## Parity, provenance and limitations

Native CPU video features match the original native framewise bank exactly.
Raw video logits exactly match same-head recomputation from those cached
features. Versus original stored temporal-run logits, max difference is1.90734863e-06.
Probability deltas below use same-head recomputation from original cached
audio/video features, not original stored temporal-run predictions. Audio is measured CPU versus the original CUDA bank:
max pooled-feature difference1.04904175e-05; max four-choice probability difference
1.27048424e-07,0 selected-ID changes. These pass the predeclared tolerances
1e-4(feature)/1e-3(head)/1e-4(probability); no tolerance tuning after execution.
Full separate-process CPU/raw replay is byte-exact for native features/logits
and all8 JSON decisions; legitimate custom-ID/video reordering is exact.

Review found a worker dependency replay gap in the immutable producer.
An independent strict wrapper binds physical dependencies in both processes
before forwards, requires the complete backend profile SHA and rechecks
afterwards. Its separate receipt binds the completed numerical replay. No
executed producer/backend/scoring source was changed to pass these gates.

This is a TRAIN functional pilot, not independent quality evidence. The
preceding observed development results remain94.85% random /84.08% near-command
correspondence, with their previously recorded uncertainties. No new blind
result, per-modality95–98%, natural image/video reasoning, optional condition
generality, sensor quality or complete Omni release is claimed.

## Actual CPU resources

| Execution | Seconds (producer) | Parent peak GiB | Worker peak GiB | Sampled concurrent GiB | Score-only p50 / p95 ms |
| --- | --- | --- | --- | --- | --- |
| primary | 157.204 | 1.804 | 4.043 | 4.542 | 0.1220 / 0.1396 |
| fresh replay | 80.915 | 1.811 | 4.047 | 4.704 | 0.1365 / 0.1617 |

Score-only latency excludes cold load, hashes, ZIP/decode, preprocessing,
native encoder and head forward. It is not end-to-end raw latency. Raw
encoding and child decode/forward/cold timings are in the small JSON/receipts.
Sampled concurrent RAM includes only this producer and its own children;
individual peaks are separate-process maxima. Windows approved system Python
uses the existing isolated JAX/TF dependency libraries; Torch remains in its
existing environment. No venv-executable policy override/dependency upgrade.
CPU i7-12700H; FP32 neural/FP64 scoring. Zero GPU/cloud/gradient/model updates.

122790899 parameters/coefficients count both full106355 heads, native
encoders122573568, two2304 normalizers,12native temperatures and1query
temperature. No dead-branch pruning/deduplication credit. This AV composition
does not count/prove a complete four-modality product, package size or phone
runtime. Weights are referenced locally; this is not a relocated release bundle.

[Exact commands, runtime, resource/feature deltas and receipt hashes](R5_NATIVE_AV_CORRESPONDENCE_RESULTS_20261015.json).
Source/native/learned/media/runtime/output inputs are verified before/after
publication; large raw media and weight/checkpoint files stay outside Git.

Next remains independent task/OOD and conditional-semantic quality coverage
before sharing/merging/compression or product adoption. Current goal is active
and unmet; preserve this runnable AV primitive as a measured reference.
