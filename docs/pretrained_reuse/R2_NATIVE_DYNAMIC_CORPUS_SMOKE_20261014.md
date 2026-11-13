# R2 native dynamic corpus — smoke admission, 2026-10-14

## Actual result and main-run status

**Native smoke PASS; full corpus not complete at this publication.**
[Fixed protocol](R2_NATIVE_DYNAMIC_CORPUS_PROTOCOL_20261014.md).
Source commit `566af2fafd5895711e2cf206615ff067014e8fa9`.
The separate smoke process completed at train10103/times0 and15. Native full
1,024 proposals per frame, all five archived arrays retained; no TopK or GT.
Frame0 exactly matches its protected parent. At both times, cached second
session chunk64–127 raw masks and scores match an independently fresh session
shape/dtype/bytes. This uses the unchanged native high-resolution wrapper,
not the nonadmitted batched-head path; prior guard failures remain intact.

| Smoke source | Full proposals | NMS survivors | Cached forward seconds |
|---|---:|---:|---:|
| 10103/time0 |1,024|55|15.587499|
| 10103/time15 |1,024|44|15.750440|

Body44.734420s after module imports. Forward columns are two full
native1,024-proposal windows; fresh-prefix replay work is outside those windows
but inside body. Unequal smoke workloads do not establish a speedup. Four
encoder calls total: two cached source frames plus two fresh prefixes.
Peak Torch allocation1,090,208,768 bytes (~1.02GiB), reserved
1,207,959,552 bytes. Frozen38,962,498 parameters/no gradients,
CPU/CUDA RNG and source files unchanged. Before/after state SHA-256:
`658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8`.
Existing RTX3080 Laptop16GiB, FP32/sdpa/TF32off/deterministic seed17/CPU4;
Torch2.6.0+cu124, Transformers5.6.2. No upgrades/downloads/cloud.

CPU preflight verified eight known train assets,256 fixed source times,
39 unique reusable frame archives and217 new inference tasks. Existing
10903/32-time corpus and all eight initial-frame arrays are reused unchanged;
legacy selected3 fields do not enter the new five-array corpus. Their full
NMS arrays are retained.8 assets/256 frames do not imply256 independent scenes
or any particular number of unique objects/QA examples.

After the passing smoke, the main command was actually started in a separate
process/output. The initial new10103/time15 full five arrays also match the
smoke byte-for-byte in a CPU audit. **This early equality is not whole-corpus
completion/replay.** Main PID38972, exec session22985, output
`C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01/`.
It reports per-frame progress; follow the same live handle without restarting.
Only terminal status, full256-row audit and separate fresh-frame replay can
establish complete input admission. Do not use a stale progress file as proof
that a process is live or finished.

## Verification and limits

34 new CPU tests: fixed8×32 plan/metadata, complete storage schema, independent
smoke mandatory, protected actual reference-config bytes. Absent API tests first
failed, then passed. Local Ruff, full CPU suite **1,255 passed/1 skipped,16.35s**,
model/dataset manifests, four catalogs and policy validation passed.
Pre-GPU review found a reference-config byte verification gap; it was fixed
with actual-file regression tests and end-of-run reference/source hash checks.
Read-only re-review found no remaining Critical/Important issues.

CPU audit (1.797749s) rechecks smoke source/config hashes,
both complete1,024 storage archives, all8 cached/fresh prefix arrays, original
frame0 five-array equality and initial main/time15 five-array equality. It loads
no model/GPU/annotations. It does not audit future main rows, full-corpus freeze
at termination, identity semantics, readout quality or held-out behavior.

No labels/dev/sealed audit/training/quantization/product promotion/main merge.
All prior Teacher/model/config/corpus/checkpoint/results protected. These are
known training scenes for a new semantic cue/cover input, not blind evaluation.
No Decision Accuracy/NLL/Brier/ECE measured here; <=200M full resident package,
text/image/audio/video/joint/OOD95–98% quality remain unmet. Do not rerun closed
nearest-distance matching or batched-head width searches.

## Saved evidence and continuation

[Small publication JSON](R2_NATIVE_DYNAMIC_CORPUS_SMOKE_20261014.json), SHA-256
`6e005e469078880cf8c3ce07c87269bbe71eeb7a0a6fe57b116173d1b3eb1324`.
Config SHA-256 `5d4417e13d8c2a4f27593d3149ac4167e7d8ee13ff9c87a076ff9780f60aa79f`;
smoke result SHA-256 `d981602a7f032048468a74e6d59416425d4f375eff482d006afc6ce8ca695681`;
CPU audit SHA-256 `5455dc4ebea4c9526cd8791feb9a44d79c53fb6528bd34c205625b4880f04ff0`.
All raw arrays/media/weights stay outside Git. Commands and fresh-only paths
are in the protocol. Main must continue unchanged; after completion:

1. Require actual terminal exit0 and complete ordered256 rows.
2. Execute the predefined separate fresh10103/time15 comparison.
3. Audit all saved arrays/parent hashes/native time/RGB binding before post-hoc
   labels. Record actual time/VRAM/output sizes and main-wide RNG/freeze checks.
4. Only then construct separately specified input-only semantic cue-to-cover /
   identity-memory supervision. Source labels must not become runtime seeds.
