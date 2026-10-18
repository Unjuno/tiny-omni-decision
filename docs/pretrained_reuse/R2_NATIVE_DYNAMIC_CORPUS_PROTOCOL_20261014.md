# R2 native dynamic proposal corpus — fixed 2026-10-14

## Purpose and limits

Semantic cue-to-cover and identity-memory learning needs later object candidates
across more than one known scene. Use **eight pinned known train clips**, each
with its existing32 uniformly sampled source times and frozen TAPIR image
features. Expand native SAM2 proposal coverage only; no QA model fit/selection,
source labels, dev/sealed data or quality claim in this extraction process.
These eight training clips are development inputs, not held-out evidence.

Fixed ordered IDs:10103,10186,10212,10387,1087,10886,10894,10903. All32 times
per clip, every1,024 positive32×32 grid prompt per time, full packed masks,
presence scores, normalized boxes, points and all NMS0.7 survivors. No TopK,
annotation seeds, source role/visibility filtering, absent-case deletion,
nearest-distance cue assignment or batch-width retry.

## Model and ancestry

Unchanged frozen native `Sam2VideoModel`38,962,498 parameters, checkpoint
`de431c4043854a71d8101e17995dfe596bf101a5`, weight SHA-256
`48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.
FP32/sdpa/TF32off/deterministic seed17/CPU4. Original high-resolution wrapper
uses64 independent single-object head calls per session,16 session chunks per
source frame. Reuse native image cache within one frame only; reset object
history between chunks; new session for each source time.

**Do not use the nonadmitted batched-head helper.** All16/32/64 score failures
stay intact. No tolerance change or old-corpus substitution follows here.

Reuse unchanged arrays from all eight initial-frame archives and complete
10903 native32-time discovery. Their intersection is one initial frame:
**39 unique frame records reused;217 new source times inferred**,256 total.
Parent archives may contain a legacy selected3 field; only the full five
proposal arrays enter this new corpus, and all NMS candidates remain. Record
parent file/hash/kind for every reused row. Do not regenerate existing results.
Hash-check each parent result/config/time bank/media/proposal and inherited
source code; conflicting existing source pins must be rejected before extension.

## Implementation and execution gates

New `native_proposal_corpus.py`, `extract_native_proposal_corpus.py`, CPU tests.
TDD: fixed ordered8×32 plan, deterministic exact times/cuts/train-only ancestry,
complete1,024 schemas/dtypes/finite values/unique valid NMS identities, and
mandatory two-frame smoke. CPU tests do not load weights/GPU.

Separate **smoke** process, first clip10103 at its first two fixed times (0,15):
full native1,024 each. Frame0 five arrays must match the original parent bytes.
At each time, second session chunk64–127 cached raw/score must match a fresh
session byte-for-byte; save/readback/hash every prefix. Two image encodes per
smoke time (one cached frame, one fresh prefix), four total. No timing speedup
claim from unequal smoke workloads.

Only after complete smoke/config/hash/freeze/RNG admission, execute remaining
217 new times with the exact same runner/config in a separate output directory.
Reused records are hash-checked/read back; freshly decoded RGB hashes are
retained for all256 rows. Save progress per frame and preserve every attempt.
Main terminal PASS requires complete ordered256 rows, all saved arrays readback
exact, one encoder per inferred frame/zero per reuse, unchanged frozen state,
flags/gradients/RNG/source hashes. No inference retries with changed settings.

After full completion, separate fresh process repeats **10103/time15**, compares
all five arrays under identical source/config hashes. Independent CPU audit
must inspect complete coverage/schema/ancestry/readback hashes/source time/RGB
binding before any post-hoc training labels. This is input admission, not QA.

Local existing RTX3080 Laptop16GiB only; GPU free≥12GiB, disk≥8GiB+512MiB
before loading. Preserve8GiB disk reserve, Torch peak allocation≤6GiB; body
cap180s smoke/7,200s full after imports. Caps are checked between calls/frames,
not hard watchdogs. OOM/nonfinite/hash/data/resource errors retain incomplete
output; never report incomplete as quality failure. No model/environment
upgrades/downloads/cloud cost/old run intervention/quantization/main merge.

## Commands

```powershell
python -m scripts.extract_native_proposal_corpus --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-smoke-01
python -m scripts.extract_native_proposal_corpus --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01
python -m scripts.extract_native_proposal_corpus --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-smoke-01 --replay C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-eight-01 --output C:/CodexArtifacts/pretrained-reuse-r2/native-dynamic-corpus-fresh-frame-01
```

Fresh paths mandatory. Code/protocol committed before GPU execution. Weights,
media and NPZ stay outside Git. Keep running job status distinct from completion.
Expected local cost roughly one hour from prior measured native32-time throughput;
report actual execution/resource data. Unique objects/semantic roles are unknown
before post-hoc labels; NMS/query counts must not be reported as unique objects.
The broader <=200M Omni task/joint/OOD/calibration objectives remain open.
