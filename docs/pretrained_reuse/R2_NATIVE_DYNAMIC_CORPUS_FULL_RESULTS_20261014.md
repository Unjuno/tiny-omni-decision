# R2 native dynamic eight-clip corpus — completed 2026-10-14

## Executed result

**PASS: native extraction, fixed fresh-frame replay and independent complete input
audit.** This is input admission for eight already-known training assets; it does
not establish semantic cue-to-cover binding, Decision quality or the Omni goal.
The original process completed naturally; no stop/restart/config/source intervention.
No model fitting, annotations, dev/sealed evaluation, cloud, quantization or main merge.

All8 clips×32 fixed times =256 ordered records/262,144 raw proposals. Reused39
existing full-five-array records and inferred217 new times. Standard native SAM2
single-object decoder path,1,024 candidates per time, unchanged mask0/NMS0.7,
no TopK/GT filtering. All NMS survivors retained: **18,441 proposals**,
not unique objects or independent examples. Total256 NPZ archives94,283,295bytes.

| Known training clip | New times | Reused times | NMS survivors |
|---|---:|---:|---:|
| video_10103 | 31 | 1 | 1731 |
| video_10186 | 31 | 1 | 3876 |
| video_10212 | 31 | 1 | 1839 |
| video_10387 | 31 | 1 | 3210 |
| video_1087 | 31 | 1 | 2189 |
| video_10886 | 31 | 1 | 2303 |
| video_10894 | 31 | 1 | 2223 |
| video_10903 | 0 | 32 | 1070 |

## Actual execution and checks

Main source commit `566af2fafd5895711e2cf206615ff067014e8fa9`; started `2026-10-14T06:45:33.891905+00:00`,
ended `2026-10-14T07:48:23.437388+00:00`. Process body3769.548180s; sum recorded
cached forward windows3612.776601s, not equivalent
to total wall time. Single native frame encode per new time/zero per reused time:
217 total. Model38,962,498 parameters, all frozen;
0 added/trainable parameters. RTX3080Laptop16GB/Torch2.6.0+cu124/
Transformers5.6.2; peak Torch allocation682,148,352bytes,
reserved836,763,648bytes. These are Torch allocator measures,
not full device/process memory. CPU tests/probes overlapped the main; no isolated
throughput, mobile/product latency or energy claim. Local cost only/cloud0.

Frozen persistent model state is identical before/after:
`658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8`. No gradients/trainable flags, CPU/CUDA RNG or
protected source hashes changed. All256 archives read back exactly; all39 parent
five-array records matched. Original config/weights/media/older artifacts remain
unchanged. Earlier failed batched/chunk decoder paths were **not** used.

Fresh separate process repeated fixed `video_10103/time15` with same config/source
hashes: all5 array shapes/dtypes/bytes exact; body25.191810s.
This is **one frame's** numerical reproducibility check, not256-frame fresh replay.
Independent CPU audit body47.599010s checks all256 full raw banks:
big-bit mask unpacking→boxes, native coordinate grid, presence gating, unchanged
NMS/ties/order/schema, source/parent/config/archive hashes. Independently decoded
all8 native videos' RGB at all256 fixed times and matched every saved RGB hash.
It rechecks the completed report and fresh-frame evidence and never promotes a
running snapshot to completion. No model/GPU/labels loaded by the auditor.

## Fixed protocol, commands and evidence

Config hash `5d4417e13d8c2a4f27593d3149ac4167e7d8ee13ff9c87a076ff9780f60aa79f`. The
[small JSON](R2_NATIVE_DYNAMIC_CORPUS_FULL_RESULTS_20261014.json) preserves report
summaries, exact run/result/config paths/hashes and per-clip counts. Complete256-row
reports, captures and NPZ remain under `C:/CodexArtifacts/pretrained-reuse-r2/`.
The [native protocol](R2_NATIVE_DYNAMIC_CORPUS_PROTOCOL_20261014.md) and
[prefix record](R2_NATIVE_DYNAMIC_CORPUS_PREFIX_AUDIT_20261014.md) are retained;
their earlier incomplete scope remains historically accurate.

```powershell
python -m scripts.extract_native_proposal_corpus --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01
python -m scripts.extract_native_proposal_corpus --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-smoke-01 --replay C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-fresh-frame-01
python -m scripts.audit_native_proposal_corpus --input C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01 --fresh-replay C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-fresh-frame-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-full-audit-01
```

Paths above record actual executions; never overwrite them or rerun an existing
primary. Local Ruff and CPU suite (1,292 passed/1 skipped,17.04s) passed; model/dataset
manifests, four catalogs and dataset policy passed. Read-only publication review
found no issues. Exact-head Actions after push is recorded in a local publication
status. No large data in Git.

## Remaining semantic and product work

This gate now permits posthoc **training-only** role/opportunity analysis and a
separately hash-bound mask-region feature bank. Native seeds/candidates must remain
input-only; source boxes/roles may supervise training, never runtime initialization.
Fix semantic cue-to-cover/identity-memory supervision and evaluation controls before
a new readout fit. No repeat of closed nearest-distance/cosine policies, no detector
ceiling inference from missing source boxes, no GT-proxy quality/adoption claim.
Text/image/audio/video joint/OOD/calibration95–98% quality and <=200M deployable
product remain unmet; native AV source rights and genuinely useful fusion still
require their own public-source evidence.
