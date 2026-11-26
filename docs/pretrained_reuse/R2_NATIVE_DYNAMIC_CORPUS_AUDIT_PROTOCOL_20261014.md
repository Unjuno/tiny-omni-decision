# R2 independent native corpus audit — 2026-10-14

The fixed native eight-clip extraction is running under its existing code and
config. This audit is separate, CPU-only and read-only for every input artifact.
No extraction code/config/weights/output is modified; no GPU/model, annotations,
dev/sealed data or training is loaded. Snapshot evidence never proves a live PID.

## Independent array reconstruction

New `native_proposal_audit.py` imports no extractor/helper/model. For all1,024
saved packed masks at each admitted record, independently reconstruct normalized
boxes, native source grid order from image dimensions, score-ordered full NMS0.7
with original-index ties, and no-object presence semantics. Empty proposals remain
in the raw inventory. Require exact dtypes/shapes/bytes and all NMS identities;
do not count NMS survivors as unique objects or hand-pick successful cases.

TDD tests exercise correct geometry, score/tie ordering, full empty inventory,
mutated masks/box/NMS/grid/presence/dtypes/nonfinite values; separate coverage tests
reject skipped/reordered/duplicated/float-time/partial-final/running-final records.

## Snapshot and full admission are separate

`--snapshot` reads current progress once and saves those exact report bytes in a
fresh audit directory. Verify the complete observed ordered prefix against the
fixed8×32 plan, every saved row file/hash/schema/geometry/NMS, correct parent
ancestry/reused full five arrays, and actual freshly decoded native source RGB
hashes/dimensions. Legitimate ongoing appends do not invalidate the captured prefix.
The prefix is marked incomplete regardless of count. Process liveness must be
checked separately through the existing tool handle/PID; do not restart a job
because an observation times out. Snapshot CPU work overlaps native extraction;
this is not an isolated throughput benchmark.

Default full admission requires terminal native extraction PASS, all256 ordered
records (39 reused/217 new), all source/code/config hashes, frozen state/RNG flags,
and the predefined separate fresh10103/time15 replay. Independently compare all
five fresh arrays with their main counterparts and bind the actual fresh config
and reference result SHA-256. Then re-decode all eight sources and check every
sampled frame's native RGB. Source events/roles/GT are still not opened.

Run normal Python with assertions enabled; optimized Python is explicitly rejected.
Input snapshot/report, auditor/helper SHA-256, all per-frame counts, full/source
RGB checks, mode, output hash, CPU body time and no-quality scope are saved. Fresh
output only; packed arrays/media/model weights remain outside Git. A failed audit
must never be converted to an admitted input by changing thresholds or rows.

```powershell
python -m scripts.audit_native_proposal_corpus --snapshot --input C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-prefix-audit-01
```

Only after native extraction and its separate fresh-frame replay complete:

```powershell
python -m scripts.audit_native_proposal_corpus --input C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01 --fresh-replay C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-fresh-frame-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-full-audit-01
```

These gates admit input provenance, not Decision Accuracy/NLL/Brier/ECE, unique
physical objects, semantic cue-to-cover correctness, joint/OOD benefit or full
<=200M /95–98% Omni completion. Full extraction/fresh replay still pending when
this audit protocol is added; the source experiment must continue unchanged.
