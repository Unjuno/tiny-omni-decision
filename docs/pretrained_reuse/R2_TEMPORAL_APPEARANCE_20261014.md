# R2: per-time spatial appearance — executed 2026-10-14

## Result and purpose

The existing frozen TAPIR checkpoint actually extracted and saved spatial
appearance at 32 predetermined times on each of eight already known train clips.
Execution, saved-cache audit and fresh-process reload passed. This establishes
input retention only: hidden-item recognition, target-to-cover binding and
Decision Accuracy/NLL/Brier/ECE were **not evaluated**. No new head was trained.

[Previous mask trajectories](R2_MASK_TEMPORAL_RESULTS_20261014.md) retained
initial appearance plus full-clip motion. A hidden item can appear later: the
[source-cue audit](R2_PERCEPTION_SOURCE_CUES_20261014.md) found video_10903's hidden
track begins at source frame224. Initial descriptors cannot encode that later
view. This probe preserves later whole-frame features without GT-selected ROIs,
questions, class prompts or annotation-dependent sampling. It does not establish
that TAPIR's tracking descriptors can recognize the item's identity.

## Fixed input and implementation

Code source commit: `5e4754a0b0f5d96dc4833f2d6c76225ab6a3b133`.
The throwaway external runner/config and exact SHA-256s are recorded in the
[publication ledger](R2_TEMPORAL_APPEARANCE_20261014.json). No production behavior
changed. Source/checkpoint revisions remain those in the
[TAPIR admission audit](R1_TAPIR_ADMISSION_20261013.md): strict legacy
`pyramid_level=0, extra_convs=False`, 31,072,327 frozen registered parameters;
no new parameters or upgrade.

Pinned raw media were decoded with PyAV to RGB256, then scaled to [-1,1].
The 32 ordered indices are inherited from the prior trajectory cache, uniformly
spanning the same official input cut. Old motion tracking still consumed all
4,979 frames; **this appearance probe encoded only 256 frames**. Full original
mask/trajectory artifacts remain unchanged. video_10903 includes sampled frame234;
it is not a source-selected reveal frame.

The official `get_feature_grids` spatial ResNet operates on each sampled frame;
`get_query_features` obtains interpolated descriptors at a fixed row-major16×16
grid at each local time. This is not a semantic detector or a new temporal encoder.
All spatial regions are retained, since a coarse grid may miss a small item.
Each clip's NPZ contains:

| Field | Actual shape | Dtype |
|---|---|---|
| lowres | 32×32×32×256 | FP32 |
| hires | 32×64×64×128 | FP32 |
| fixed_grid_descriptors | 32×256×384 | FP32 |
| frame_indices | 32 | int64 |

The upstream dataclass docstring describes 64 hires channels; this pinned legacy
checkpoint/constructor actually emits128. Shapes were measured, not inferred
from that stale description. Low/hi spatial vectors are unit-normalized.

## Predeclared checks and actual outcomes

Initial mask-query descriptors obtained from the sparse grids must agree with
the saved full-tracking initial descriptors within max absolute difference1e-5.
Same-input sparse extraction must repeat bitwise. These gates were declared
before inference and were not relaxed.

| Check | Actual outcome |
|---|---|
| Initial-descriptor max absolute difference, all clips | 1.16787850856781e-6; passed1e-5 gate |
| First clip consecutive sparse grids/descriptors | bitwise equal |
| CPU/CUDA RNG and query inputs | unchanged |
| Frozen model state before/after | SHA-256 equal; no gradients |
| All eight saves/readbacks | finite, shapes/times/hash verified; bitwise equal |
| Fresh constructor/checkpoint process, first clip | lowres/hires/descriptors bitwise equal |
| Independent CPU cache audit | all eight passed |

State SHA-256: `3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a`.
No annotations, validation feature arrays or sealed audit were opened by these
three processes. One extraction, one CPU audit and one fresh reload were executed;
no failed attempt or additional seed search.

Later descriptors differ from t0 for all63,488 comparisons out of65,536 total
descriptors. Mean L2 distance from t0 over later times:

| Known train clip | Mean L2 |
|---|---:|
| video_10103 | 1.217703 |
| video_10186 | 1.411729 |
| video_10212 | 1.489433 |
| video_10387 | 1.355489 |
| video_1087 | 1.284806 |
| video_10886 | 1.423115 |
| video_10894 | 1.349154 |
| video_10903 | 1.316401 |

These are correlated numerical retention checks, not semantic quality metrics.
Motion, backgrounds or irrelevant objects can also cause variation. No hidden
target was identified, no deployable cover selector was produced, and no quality
claim follows from these differences. Initial-broadcast comparison was computed
against cached t0; no duplicate broadcast cache was written.

## Resources and artifacts

RTX3080 Laptop16GB; Python3.11.9, Torch2.6.0+cu124, NumPy2.4.6, PyAV18.1.0.
FP32/TF32off, deterministic, seed17, four CPU threads; same local environment.
Main body65.421269s; synchronized encoder/descriptor windows sum3.274663s.
Peak Torch allocation715,315,200bytes, reservation876,609,536bytes. Fresh-process
forward/compare0.744768s, peak allocation576,633,856bytes. Imports and driver
memory are excluded; these are not complete Decision latency or exact GPU time.
CPU peak RAM unmeasured. Cloud cost0; no paid compute or training.

Saved arrays total905,971,712bytes uncompressed; compressed NPZ841,255,778bytes.
This is a research cache cost, not a model package size. TAPIR weights remain
124,348,474bytes; inherited SAM2+TAPIR weights280,256,538bytes already exceed the
provisional200MB package gate before other Omni components. No quantization.

Local artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:
`perception-temporal-appearance-01/`,
`perception-temporal-appearance-audit-01.json`,
`perception-temporal-appearance-reload-01/` and the three executed Python runners.
Config/result/media/source/script/NPZ hashes and exact commands are in the ledger.
Run result SHA-256 `a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4`;
config `b86589ce82c5b1bde7657ffa4b1f80abc5464e54f63211a3f2fd728fff774a79`;
fresh reload result `c60fad4a28b24aff30e22bfbc829ee553ac1a7231c628447c23f2a2700ca79f6`.
Weights, media and large arrays remain outside Git. Legacy inputs were read only.

Publication verification: Ruff passed;978 CPU tests passed and1 skipped in15.79s.
Model/dataset manifests, all four catalogs and the example policy audit passed;
log `C:/CodexArtifacts/pretrained-reuse-r2/temporal-appearance-manifest-validation-01.log`.
The executed external GPU/source checks above remain outside normal CPU CI.

## Next quality gate

Do not scale training merely because feature retention passed. Define an
input-only supervised target-association task before fitting: source roles/events
may supervise losses, but GT identities/boxes cannot select inference inputs.
Preserve variable-sized groups and actor-relative semantics; `is_masked=0` alone
is not a verified visible-target cue. Fix split/selection/control rules before
measuring whether later features help target binding or conditional decisions.
The existing known development subset is not blind evaluation. Four-modality
quality, genuine joint benefit, OOD robustness and the final package remain open.
