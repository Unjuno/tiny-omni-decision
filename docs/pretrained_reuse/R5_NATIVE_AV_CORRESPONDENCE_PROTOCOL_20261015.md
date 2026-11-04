# Native AV correspondence execution and state reuse

Previous goal turn is progress: actual symmetric correspondence/replay and exact
HEAD CI at c65ae58. Product scope remains <=200M all resident prequantization
parameters, native four-modality decisions/custom JSON and task95–98%/OOD gates.
This bounded runtime extension preserves the measured four-choice AV task and
all existing models/configs/corpora. User permits autonomous local/public work.

## Proposed execution, fixed before new forwards

Use immutable audio24 and video_ordered18 temporal heads, train-only normalizers,
six native temperatures and query temperature1.0. No learning, fitting, audit
access or score-based sample choice. First train anchor's frozen random and hard
quartets supply all four query roles each. All distinct members' actual audio
and front video archive bytes are decoded; no label/transcript/source ID enters
an encoder/head/scorer. Parent metadata is solely for media/hash/parity checks.

Load the existing exported8,208,384 Whisper encoder without its generation decoder.
CPU4threads/FP32, official16kHz frontend/full30s padding and mean of complete
physical ceil(samples/320) tokens. Original bank was CUDA: prospective CPU-vs-bank
gate max absolute pooled-feature difference1e-4, raw head difference1e-3,
four-choice probability difference1e-4, no changed selected IDs. Differences are
cross-backend tolerance results, not claimed byte-exact CPU/CUDA equivalence.

Use actual native JAX/TF VideoPrism114,365,184 with its immutable checkpoint/source
and already installed isolated dependency libraries, via approved system Python.
No dependency upgrade, new executable installation or control-policy change.
Decode every video frame/PTS, nearest uniform actual first-to-last16 selection,
TF bilinear short288/center288,4096 tokens->16x256x768 spatial means. Separate CPU
process isolates incompatible NumPy/Torch/JAX stacks. Frame features must exactly
match existing native bank for every selected member. Fail if not; no tolerance
increase, preprocessing or sample reduction after failure.

The runtime consumes raw bytes via injected, caller-owned verified frozen modality
backends which return51 native raw logits. Cache/state key binds full media SHA256,
modality, full frontend/head/calibration profile SHA256 and supplied time range.
Immutable owner-bound logits, bounded16entry LRU; unknown/changed profile or bytes
cannot reuse stale state. Four opaque unique option IDs only, full semantic options
retained, exact original agreement/softmax/custom-ID JSON. Reordered IDs+videos
must yield identical JSON. Missing a required stream uses only the documented
uniform prototype policy; decoding errors never silently become missing evidence.
This task-specific51logit state is not a general world-state representation.

Measure cold load/decode/encode/readout/first decision, every repeated decision
and cache hit/avoided actual encoder call, payload size and each process peak RAM.
Cross-process total RAM needs observed simultaneous process sampling; sum of
independent maxima is a conservative envelope, not observed simultaneous peak.
Count both full heads/normalizers and native encoders:122790899 parameters and
coefficients, AV slice only. New state/JSON/source metadata only, no weight copies.
CPU run cap600s, each process6GiB peak, new output64MiB, disk reserve1GiB; old
GPU8GiB guard and unfinished run stay untouched. Existing other processes protected.

All native/learned/raw-media/source/runtime inputs are SHA-bound before/after.
Source is committed before execution. Fresh-process repeat reloads models and raw
media, requires identical CPU outputs and JSON/counter accounting, preserves all
attempts. Test pure runtime ownership/cache/validation/JSON/reference math first.
Full CPU/NoTorch tests/Ruff/manifests and exact pushed HEAD CI before publication.
Report functional parity separately from quality: no new blind measurements,
general natural-video/sensor quality,95–98% or complete Omni release claim.
