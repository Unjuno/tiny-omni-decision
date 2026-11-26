# STARSS23 fixed-window Decision input probe

This bounded CPU probe follows the source acquisition record. It tests real
joint-input preparation and label semantics, not model quality. Existing model
weights, training runs and datasets remain untouched. No dev-test or official
evaluation content was read.

## Fixed choices before execution

- Use both previously acquired dev-train recordings; choose `[0,5)` and `[5,10)`
  seconds from each without inspecting labels to pick interesting intervals.
- Preserve all four audio channels at the original 24 kHz. Decode PCM16 to
  float32 by division by 32768; do not average directional FOA channels.
- Select eight RGB video frames per window, at the first presentation timestamp
  at or after each equal-width bin midpoint. Reject delays above 50 ms and
  timestamps outside the window. Keep equirectangular pixels unchanged.
- Use upstream 100 ms annotation ticks in half-open windows. A question's target
  is yes when at least one annotation tick of that event occurs in the window.
  Preserve overlapping events as independent binary queries. Count distinct
  active ticks, not overlapping source instances, when recording duration.
- Merge female/male speech annotations into generic speech. Emit all twelve
  event queries for every window; do not discard negative queries or filter
  difficult/ambiguous events. The explicit task is annotated *prominent* event
  presence, not exhaustive audible-event detection.
- Every window/query inherits recording and room identity. All four windows
  remain dev-train. Two rooms cannot establish useful held-out quality.

## Execution and measured results

Source base: `73255d9bccf99ec49cd0b24d92d8a017e94d73bc` plus the new runner,
whose exact SHA-256 is saved in the result. Command:

Environment: Windows x86-64, Python 3.11.9, PyAV 18.1.0, NumPy 2.4.6;
CPU-only processing, no GPU allocation or paid compute.

```powershell
python scripts/probe_starss23_windows.py --source-result C:\CodexArtifacts\pretrained-reuse-r2\starss23-paired-smoke-01\result.json --output-dir C:\CodexArtifacts\pretrained-reuse-r2\starss23-window-smoke-01
```

PASS: four windows and 48 binary event queries. Each decoded audio array has
shape `[120000,4]`; each video array `[8,960,1920,3]`. All arrays passed exact
save/reload checks. Source media and annotation hashes were verified first.
There were five positive labels: speech in all four windows, laughter in one.
This prevalence makes aggregate accuracy particularly misleading; no accuracy
was computed and no readout was fitted.

The four compressed arrays total 68,998,092 bytes and remain outside Git.
`R2_STARSS23_WINDOWS_20261011.json` preserves timestamps, identities, query
labels, local output hashes and source-result hash. Its SHA-256 is
`7ed1bf595a36ae9dd334c191806bad8e8d2b9f4574f051c5a40b2f8b79f4c3f1`.
Tests cover half-open boundaries, simultaneous events, merged-speech duplicate
ticks, malformed annotations, empty windows and dev-test rejection. Full CPU
suite: 261 passed, one skipped; Ruff passed.

## Unresolved before quality fitting

Follow-up: `R2_STARSS23_FEATURES_20261011.md` records successful real frozen
audio/video extraction and repeated-forward parity for these exact windows.
The statements below describe what was unresolved at the window-probe stage;
fusion/readout fitting and independent quality evaluation are still outstanding.

Real input and target plumbing now works, but synchronized latent extraction,
fusion/readout fitting and matched unimodal/AV ablations are still unmeasured.
The preserved four-channel audio needs an explicitly specified mono or spatial
encoder input; equirectangular frames need a fixed visual preprocessing rule.
Do not silently reuse whole-recording features for these five-second labels.
Before any quality run, freeze adequate room-disjoint train/calibration/
validation partitions and report per-event balanced metrics alongside
Accuracy/NLL/Brier/ECE. Negative prevalence, room count and speaker-identity
uncertainty must be reported. No final product threshold is implied by this
plumbing result.
