# Path A typed Decision JSON dispatcher — 2026-10-22

## Change

Path A's existing text, image, audio, and video scorer responses can now be
wrapped in the typed Decision JSON v1 contract. The caller supplies the stable
request ID and option IDs/text in the same order as the existing probability
vector. The wrapper validates and serializes those values; it adds no model
parameters, calibration, training, checkpoint changes, or new scoring logic.
The legacy raw dispatcher remains available.

The CPU-only replay was made against the existing development validation
inputs and an isolated package view. It loaded no sealed audit and performed no
training. Source artifacts and frozen validation data were read-only.

## Replay result

Attempt directory:
`C:\CodexArtifacts\pretrained-reuse-r2\path-a-dispatcher-v0-attempt-04-typed-json-cpu`

The run used the existing reference parity threshold of `1e-5`. Video, Image,
and Text passed that comparison. Audio retained all four predicted classes and
its JSON records matched the runtime probabilities, selected option, IDs, and
ordering. Audio's maximum absolute probability delta from the saved reference
was `0.000617743`, exceeding `1e-5`; therefore the CPU replay did not pass the
saved-output numerical parity gate. This is reported as a reference-parity
failure, not as a typed JSON failure or a quality result. The cause of this
CPU/reference probability difference is not established here.

The saved attempt predates separating those two status fields in the harness.
Its item-level evidence shows typed JSON consistency for each modality; future
reports expose `typed_json_status` independently from saved-output `status`.
The attempt remains preserved and was not rewritten.

## CI follow-up

The first push passed lint but failed 3 tests in
`tests/test_observation_update_probe.py`: two existing raw-dispatch probes call
the shared comparison helper without typed JSON metadata. The helper now treats
typed JSON checks as optional and runs them only when typed decision specs are
provided; raw reference parity behavior remains unchanged. After this
compatibility fix, the complete local CPU suite passed: 2,380 passed, 11
skipped. Ruff and `git diff --check` also passed. The resulting CI run is being
checked separately after push.

## Verification limits

Targeted CPU tests cover the JSON schema, validation rules, runtime dispatcher
mapping, audio multi-query mapping, and existing decision behavior. This is an
interface-contract check only. It does not establish calibration, accuracy,
cross-modal reasoning, sensor support, or product readiness. No GPU was used.

