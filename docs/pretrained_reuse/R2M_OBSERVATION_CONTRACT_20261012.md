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
