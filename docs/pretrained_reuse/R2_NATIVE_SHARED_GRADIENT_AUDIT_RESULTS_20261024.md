# Shared native readout gradient audit — 2026-10-24

## Execution result

**PASS — TRAIN-only diagnostic completed.** This was gradient measurement on a
frozen model, not a fit or a benchmark evaluation. The model had zero optimizer
updates and zero weight changes. Validation and sealed audit inputs were not
loaded.

The reference is the completed repeat-coverage fit at
`C:/CodexArtifacts/pretrained-reuse-r2/native-coverage-repeat-fit-01`. Its
selected checkpoint was step 2,048, and its SHA-256 remained
`94d4e452314d0f41f43c4f1711711837c38b60e9eea7816c2149ad3636d888f2` after the
diagnostic. The original sampler SHA-256 is
`23b049b25495dc2d7a8e983b1abb3b257399413c975b2901ed8c7af171073797`.

One preliminary attempt (`native-shared-gradient-audit-01`) passed input
preflight but stopped on its first backward call because modality-inactive model
parameters return `None` gradients. It performed no optimizer or weight update
and produced no gradient measurements. The error, preflight, and IDs remain in
that separate attempt directory. The runner was corrected to represent unused
parameter gradients as zero vectors; nine targeted tests passed before retry.
The successful retry used the same frozen sample in a new directory,
`native-shared-gradient-audit-02`.

## Frozen sample and inputs

The run selected 32 optimizer-step locations from the original repeat sampler,
including the first and last updates. Each selected update contains two examples
from each of text, image, audio, and video, matching the original eight-example
accumulation group. The 256 presentations comprise 253 distinct row IDs and
three repeats, retained as sampled. They contain 251 observations, 179 media
assets, 59 video scenes, and 215 modality/group pairs.

| Modality | Presentations |
| --- | ---: |
| Text | 64 |
| Image | 64 |
| Audio | 64 |
| Video | 64 |

Source counts are CLEVR-4 64, CLEVRER 64, Lombard 37, STARSS 27,
condition-relevance text 23, curriculum text 13, original text 3, and
programmatic-decision text 25. The full source and question-type counts, sample
IDs, observation identities, and file hashes are in the pinned manifest and
`preflight.json`.

The pre-registered row-ID order SHA-256 is
`82f4e8ff288b4e29faf52c35b8c15c7a9e9c51a3cfbbd1bbd7af06b0cbdad11e`; the
selected-occurrence SHA-256 is
`3e1293e12f7e3d5c8d2ccabe6bc253a3c296d1f28beeff8882d73117e3f82dfb`. The
manifest SHA-256 is
`fc9dda42df0a683cbbb1bdb4e834137eba885491419802543e995974adb17cf7`.

### Source provenance discrepancy

The fit receipt reports source commit `b3b7bb3abc4d27b6da51aa05032d2ff14704362f`,
but five validated runtime dependency files do not match the versions at that
commit. The diagnostic bundles the exact five byte sequences whose SHA-256
values match the fit receipt. Their historical Git commit is **unknown**; the
report does not claim that the commit alone recreates the original run. See the
[source snapshot note](source_snapshots/native_coverage_repeat_fit_01/README.md).

## Gradient results

For each sampled update, the audit averaged the two per-example losses for each
modality, computed four full shared-readout gradient vectors, and compared all
six modality pairs. The original fit averages eight losses and clips the
combined gradient at norm 1.0; the clipping calculation below uses that same
`sum(g_m) / 4` scale.

| Pair | Mean cosine | Median | SD | Range | Negative fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text–image | 0.0002 | -0.0136 | 0.1390 | [-0.4277, 0.2995] | 56.3% |
| Text–audio | 0.0009 | -0.0126 | 0.0792 | [-0.1495, 0.2154] | 62.5% |
| Text–video | -0.0084 | -0.0145 | 0.1230 | [-0.2587, 0.3793] | 59.4% |
| Image–audio | 0.0085 | 0.0067 | 0.0978 | [-0.1747, 0.2498] | 46.9% |
| Image–video | -0.0150 | -0.0010 | 0.0981 | [-0.2005, 0.2356] | 50.0% |
| Audio–video | 0.0456 | 0.0489 | 0.1323 | [-0.2100, 0.3010] | 31.3% |

Mean pairwise cosine was close to zero relative to the observed spread. Negative
directions occurred in roughly half the sampled updates for most pairs. This
does not show a stable, one-directional modality conflict; it shows variable
local alignment in this selected checkpoint and fixed TRAIN sample.

| Modality | Mean gradient norm | SD | Mean projection share on combined direction |
| --- | ---: | ---: | ---: |
| Text | 2.950 | 2.463 | 0.211 |
| Image | 5.147 | 1.979 | 0.562 |
| Audio | 1.848 | 2.424 | 0.094 |
| Video | 1.980 | 0.971 | 0.132 |

Projection shares sum to approximately one per update and describe this sample
only. Image gradients tended to contribute more to the combined direction under
the original equal-example sampler; this alone does not justify changing loss
weights or adding modality-specific branches.

| Combined-gradient measure | Result across 32 updates |
| --- | ---: |
| Mean norm before clipping | 1.801 (SD 0.659; range 0.977–3.784) |
| Updates exceeding norm 1.0 | 31 / 32 (96.9%) |
| Mean clip factor | 0.622 (SD 0.203; range 0.264–1.000) |
| Mean cancellation ratio `||sum(g_m)|| / sum(||g_m||)` | 0.603 (SD 0.070) |

The fit's global clipping threshold would therefore have been active for nearly
all sampled updates at this checkpoint. This is a reason to examine gradient
scale and clipping in any next controlled optimization comparison. It is not
evidence that clipping caused the observed quality trade-offs: the audit used
one selected checkpoint, one seed, and development TRAIN examples, with no
intervention arm.

## Runtime and integrity

- Local GPU: NVIDIA GeForce RTX 3080 Laptop GPU; HotUGPU reserved 3,200 MiB and
  recorded 13,015 MiB free for the child.
- Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6; float32 eager
  `MultiheadAttention`.
- Child runtime: 7.72 seconds; peak allocated CUDA memory: 226,318,848 bytes.
  Cloud compute and cost: none.
- HotUGPU wrapper SHA-256:
  `59b038dcd61317263cb972b3098ecf99b6b3fa111c6bc2d0ffe752fc93c5a06a`;
  gate validator SHA-256:
  `5659e8694916ee2f2c62eff1c34bad3e2aba7fa32fe27bc5eefa58fe7698d3d3b`.
- The audit verified the original best checkpoint, sampler, selected TRAIN
  rows, corpus/query receipts, and all sampled feature-bank hashes. The
  selected checkpoint hash was rechecked after execution and still matches.
- A PyTorch warning said `expandable_segments` is unsupported on this Windows
  build; the diagnostic completed with the recorded memory use.
- At the post-run check the RTX 3080 showed 0 MiB used and 0% utilization. No
  Orbitune process was stopped or modified.

The structured `result.json`, `preflight.json`, exact 32-row
`measurements.jsonl`, and selected occurrences are committed beside this
report. Their hashes are respectively `870a002f68428a9559a84082ed03dba0c7656370188fbef55648ed1eaaa11c6a`,
`4410e83938105fa85f343cad235bcb62abfef1f73bf6d9d8670d16a44002fbc4`,
`21efbc431549763570b1eb562dc32f5be2d3971d403425daa761f2c1b1002ff6`, and
`2e3720ab13b7a36fababcad2f3349a13e6b979627867a02523b56179f00ff776`.

## Decision

The result is **mixed/uncertain**, not evidence to reject the shared readout or
to add separate modality heads. It finds substantial norm imbalance and very
frequent clipping, while pairwise directions average near zero and vary widely.
The next useful comparison, if pursued, should isolate the effect of the
existing clipping/gradient scale while holding the sampler, checkpoint
initialization, and evaluation policy fixed. Do not tune weights from these
TRAIN gradients or interpret this as validation or sealed-audit quality.
