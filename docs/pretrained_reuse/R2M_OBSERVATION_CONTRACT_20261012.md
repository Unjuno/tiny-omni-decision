# R2-M observation contract and sensor preprocessing

## Implemented boundary

`tiny_omni_decision.observations.ObservationBatch` is a version1.0 validated
metadata contract. Text is optional. It supports audio/image/screen/video/sensor
and text observations, stable source and option IDs, optional typed/text/spoken
criteria, source clocks, sample/token timestamps, boolean validity and hashed
payload references. An optional nonnegative payload_index identifies an item in
a batched payload. Sensor channels declare units, scale, offset and optional
valid range. Calibration describes raw-to-declared-unit affine conversion; this
metadata module does not apply calibration or normalize values.

Absent modalities are omitted; an all-invalid present stream remains present.
Fusion callers must explicitly require a shared clock or perform a documented
alignment transform. Spoken criteria must refer to a present audio source.
No text encoder, tokenizer or Torch import is required. JSON serialization uses
Pydantic, with no language generation. Existing DecisionOutput and caches remain
unchanged. This is an input contract, not a newly trained Decision model.

The module validates metadata only. It does not resolve paths, verify referenced
payload bytes/shapes/index bounds, establish synchronization, or prevent a future
loader from using target columns. Those runtime boundaries must be implemented
and tested before native inference can be claimed. Candidate IDs are opaque;
attributes carry typed meanings. A model must not infer meanings from ID strings.

## Sensor window policy and real smoke

`contiguous_windows` uses explicit timestamps and boolean validity. Windows cannot
cross an invalid sample or a time gap larger than the configured limit. Duplicate,
reversed or nonfinite times fail closed. No resampling/interpolation is performed.
This preserves time discontinuities rather than treating all50Hz source samples
as consecutive. Window length128, stride64, maximum gap30ms were fixed for the
bounded preprocessing smoke, not tuned from task quality.

The immutable official HARTH22-person archive and predeclared subject split were
verified by SHA256. Only first training subjectS025 was read; calibration,
validation and sealed-audit members remain unopened. Its231,729 samples have
22 gaps above30ms. The processor emitted3,593 windows of128x6 acceleration values
in g. All330 mixed-label windows were retained, with full label sequences saved
separately; no easy-example selection or majority-label simplification occurred.

Feature NPZ contains only values and timestamps. Teacher/supervision NPZ is
separate. Save/reload is exact. The request identifies payload_index0 explicitly.
The naive source date is preserved as an unspecified-timezone clock origin, not
silently treated as UTC or synchronized across subjects.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/harth-native-window-smoke-02`.
The first attempt is preserved: its feature preprocessing passed, but its example
request lacked an explicit batch item index; do not use that request for runtime.
The second attempt adds that index and preserves identical feature/target hashes.
[Current measured result](R2M_SENSOR_WINDOW_SMOKE_V2_20261012.json),
[initial attempt](R2M_SENSOR_WINDOW_SMOKE_20261012.json).

Command: `python -m scripts.prepare_harth_native_smoke` (refuses existing output).
Approximately1.60s CPU preprocessing; no model inference, training, calibration,
parameter increase or GPU work. This is not sensor Accuracy or fusion evidence.

## Tests and remaining M-0 work

19 focused CPU tests cover five non-text modality fixtures, JSON roundtrip,
source/option/mask validity, finite/increasing time, sensor units, missing versus
invalid presence, synchronized A/V metadata, spoken reference validity,
cross-clock rejection, encoder-free import, explicit payload index and windows
that cannot cross gaps/invalid samples.

Real raw-media adapters for this contract, native probability readouts, sensor
stem, screen dataset, event-aligned multi-signal sources and intervention evaluator
remain incomplete. No M-1 or M-2 quality gate is closed by schema fixtures.

## Verified sensor payload loader

`load_sensor_payload` now resolves the sensor NPZ path, reads bounded bytes once,
checks SHA256 against those same bytes, restricts members to values/timestamps,
and enforces compressed/decompressed byte budgets with pickle disabled. It checks
window index, dimensions, channel count and exact timestamps before returning
values. Valid samples receive the declared affine calibration and optional range
check; invalid samples return zeros **and an explicit false mask**. Output arrays
have read-only flags. Unit labels are preserved; no implicit unit conversion.

Eight tests reject changed hashes, times, indices, channel metadata, target fields
and budget overflow, and check mask/calibration behavior. The saved S025 request
loaded128x6 real values without importing Torch/Transformers: timestamps7540–10080ms,
all128 valid. One CPU load took0.0921s including reading/decompressing the batched
file; this is not per-request production latency or model inference timing.
[Measured loader result](R2M_SENSOR_PAYLOAD_LOAD_20261012.json).

The loader supports this explicitly defined sensor NPZ format only. It does not
establish learned sensor decisions, cross-modal alignment, audio/image/video
adapters, or sensor quality. The target/supervision file is rejected as input;
future runners must also avoid passing source IDs or filenames as learned features.
