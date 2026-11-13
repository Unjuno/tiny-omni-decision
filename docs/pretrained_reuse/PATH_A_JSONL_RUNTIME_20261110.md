# Path A task-scoped JSONL runtime — 2026-11-10

## What this adds

The existing four-component Path A assembly can now be called through a
single stdin/stdout JSONL interface:

```powershell
python -m tiny_omni_decision.cli path-a-jsonl `
  --assembly 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-clean-copy-20261026-01' `
  --videoprism-python 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  --videoprism-source 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
  --worker-log-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-jsonl-runtime-smoke-20261110-attempt-01' `
  --device cpu
```

The runtime verifies the assembly manifest, starts the existing workers lazily,
keeps stdout machine-readable, returns one typed Decision JSON per supplied
request, and writes worker diagnostics to the caller-provided empty log
directory. Invalid lines return a JSON error record and do not prevent later
lines from being handled. Image, audio, and video requests must supply the
asset SHA-256; the existing workers verify the actual bytes against it.
Use an assembly whose file set exactly matches its manifest. The default v4
directory currently has two unmanifested `.pyc` files and is correctly
rejected; the clean copy shown above passes verification without modification.

## Frozen task scope

The API requires a `task_id` to prevent a task-specific scorer from being
presented as a general modality model. Supported IDs map only to the packaged
development scopes:

| `task_id` | Runtime route | Scope |
|---|---|---|
| `typed-decisions-synth-text-v0` | text | Synthetic Typed Decisions Synth only |
| `path-a-image-videoprism-clevr4-runtime-v0` | image | Static CLEVR-4 taxonomy choices |
| `path-a-image-videomaev2-clevr4-100k-runtime-v0` | image_videomae | Static CLEVR-4 taxonomy choices using the pinned VideoMAEv2 Small encoder and frozen 100k readout |
| `path-a-audio-multiquery-runtime-v0` | audio | Four Speech Commands query templates |
| `path-a-audio-speech-commands-m1-v0` | audio_keyword | Ten fixed Speech Commands keywords with a pinned text-free readout |
| `clevrer-videoprism-binary-feature-decision-v0` | video | CLEVRER candidate-statement binary score; options must be ordered `wrong`, `correct` |

The M1 route adds a separate 3,850-parameter frozen readout and train-only
calibration scalar; it reuses the resident Whisper-Tiny encoder. It does not add
joint audio-video support or make task-specific scores general-purpose or
release-qualified. The four-component v4 assembly remains 170,499,716 unique
parameters and 687,254,772 bytes. Adding the separate VideoMAE image route and
the M1 readout yields a simple union of 192,709,534 parameters and 731,819,316
bytes, still above the provisional 200 MB package target.

The VideoMAE image route is a separately pinned task-specific path. It is
selected only by its explicit task ID and requires explicit encoder, readout,
and dependency paths. It does not start the VideoPrism image worker. This
route is static-image taxonomy scoring only; repeating one image across the
encoder's 16-frame input does not provide temporal evidence. Its integration
smoke and limitations are recorded in
[`PATH_A_VIDEOMAE_JSONL_IMAGE_RUNTIME_20261110.md`](PATH_A_VIDEOMAE_JSONL_IMAGE_RUNTIME_20261110.md).

## Request example

Each input line has a stable task ID, an existing worker payload, and one typed
decision specification (audio uses one per query):

```json
{"request_id":"smoke-001","task_id":"typed-decisions-synth-text-v0","payload":{"state":"A blue object is present.","question":"Which color is present?","options":["blue","red"]},"decision_specs":[{"request_id":"smoke-001","decision_type":"choice","options":[{"id":"blue","text":"blue"},{"id":"red","text":"red"}]}]}
```

The response has `status`, `request_id`, `task_id`, a validated `decision`
record, measured runtime fields, and—when a worker starts—its model revisions,
unique parameter count, and memory metadata. JSON schema validity is not
evidence of calibration or task accuracy.

## Verification status

Five CPU unit tests cover text routing, multi-query audio response shape,
mandatory media hashes, frozen video option order, and recovery after malformed
JSONL input. Separate CPU and CUDA end-to-end smokes cover text, image, video,
audio multi-query, and the new fixed-taxonomy keyword route. They are
protocol/load checks, not new quality measurements.

### Local CPU end-to-end smoke

- Task: `typed-decisions-synth-text-v0`, one synthetic protocol request,
  request ID `runtime-smoke-002`.
- Result: valid typed JSON returned; the response selected the supplied `blue`
  option with probability `0.9829811`. This single synthetic query is not an
  accuracy or calibration result.
- Device: CPU; no CUDA model execution, training, validation corpus or sealed
  audit access.
- Worker startup: `6.9218 s`; worker inference: `0.03857 s`; complete
  dispatcher request: `6.9760 s`; outer command wall time: about `9.93 s`.
- Assembly: the normal v4 directory was rejected before worker startup because
  it currently contains two unmanifested `.pyc` files under the Image
  component. It was left unchanged. The preserved clean copy has the same
  manifest SHA-256 `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`,
  and all 42 manifest-listed files passed the existing verifier.
- Receipt: `C:/CodexArtifacts/pretrained-reuse-r2/path-a-jsonl-runtime-smoke-20261110-attempt-02/run-metadata.json`,
  SHA-256 `cf839a760ce766e1826ba2cd2b79a43bcc82085bb4efd3e0dad55ce78d349f2e`.

This verifies one modality's request-to-weighted-response path. Image/audio/
video dispatch was not run end to end in this original smoke, and no joint AV
support or product-quality conclusion follows. Later route-specific smokes are
recorded below and in their linked reports.

### CUDA VideoMAE image JSONL parity smoke

On 2026-11-10, a separate train-only request exercised the new explicit
VideoMAE image route on the RTX 3080 Laptop. The actual typed probabilities
matched the pinned frozen-feature/readout reference exactly (maximum delta
0.0); peak allocated CUDA memory was 240,896,512 bytes. This validates the
integration and deterministic score path for one training image only. It does
not measure accuracy/calibration, validation, video understanding, or joint
modalities. See the linked report for pinned artifact hashes, timings, and the
immutable metadata-reconciliation note for the attempt-02 directory.

### CUDA Speech Commands M1 JSONL smoke

On 2026-11-11, the final four-route integration branch ran one SHA-verified
TRAIN WAV through the CUDA Path A JSONL CLI and returned a finite ten-option
Decision JSON in caller-specified option order. The record was a training
example, so its single correct prediction is protocol evidence only. The
worker reported 55,990,669 resident unique Torch parameters, including the
3,850-parameter M1 readout; 233,796,608 CUDA bytes were allocated after the
request. See
[`R2_SPEECH_COMMANDS_M1_JSONL_RUNTIME_20261111.md`](R2_SPEECH_COMMANDS_M1_JSONL_RUNTIME_20261111.md)
for exact source hashes, timings, artifact paths, and limitations.
