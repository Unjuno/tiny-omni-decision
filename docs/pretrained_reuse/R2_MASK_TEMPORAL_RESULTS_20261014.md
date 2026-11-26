# Model-mask temporal state: executed, quality remains open

## Actual result

Eight known training clips completed with exit0 and
`PASS_ALL_MODEL_MASK_TRAJECTORY_EXECUTION_ONLY`. All696 existing SAM2 NMS masks
seeded frozen TAPIR queries;4,979 original frames were processed without temporal
thinning, GT initialization, TopK or confidence filtering. Raw tracks, initial
appearance, ordered tokens and per-clip CPU/CUDA RNG states were saved.

Separate source scoring found24/24 initial group-object opportunities with bbox
overlap AND an own query inside the GT object, versus23/24 for fixed-grid pooling.
With **post-hoc GT-assisted initial association**, tracked points remain in future
GT boxes on371/500 annotation entries(74.2%), versus177/500(35.4%) for the same
initial points held static. All24 targets and500 eligible entries remain,
including masked source entries and uncertain predictions. These correlated
entries are not500 independent examples or hidden-item QA Accuracy.

| Train video | Masks | Distinct initial xy | Group objects | Tracked point in box | Static |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10103 | 55 | 53 | 2 | 26/30 | 23/30 |
| 10186 | 186 | 172 | 3 | 35/65 | 13/65 |
| 10212 | 43 | 40 | 3 | 91/105 | 13/105 |
| 10387 | 168 | 160 | 4 | 37/70 | 16/70 |
| 1087 | 67 | 64 | 3 | 41/57 | 51/57 |
| 10886 | 64 | 59 | 3 | 53/64 | 22/64 |
| 10894 | 73 | 66 | 3 | 36/55 | 19/55 |
| 10903 | 40 | 36 | 3 | 52/54 | 20/54 |

The696 mask queries have650 distinct xy positions summed within clips, not696
unique physical objects.1087 is worse than static and is retained. No model
training, Decision metrics, candidate adoption or95–98% Omni claim occurred.

## Implementation, review and preserved failures

`mask_trajectory_features.py` provides initial mask/point membership, per-mask
point-history means and foreground query generation; adds0 learned parameters.
All masks remain in input order. Unsupported pooling produces zero tokens with
explicit false flags/count0. Overlapping support is retained, outside points
are never clipped into a mask, and temporal columns remain ordered. Support is
not track validity; the existing readout rejects all-invalid input.

Real CPU pooling retained696 masks:487 supported,209 unsupported. Manual means,
time reversal and readback matched bitwise. Separate scoring found one glass
without a fixed-grid query. Direct mask queries require new inference; old
trajectories were not invented or relabeled. Every nonempty mask now supplies
its nearest-to-COM foreground pixel, with exact integer distance/row-major ties.
Every FP32 query was checked inside its own mask before GPU execution.

TDD first failed on absent APIs. Independent review found one Important centroid
tie issue and two Minor issues: rounding across cell boundaries and complex casts
discarding nonfinite imaginary parts. Five regression failures reproduced them.
Overflow-safe exact integer distance, exact binary-rational floor/direct equal
rasters, and rejecting non-real inputs fix them.25 focused tests pass; follow-up
review found no remaining issues and checked511 nonempty3×3 masks and the65,536
pixel overflow fallback. Plan01 remains preserved; corrected plan02 alone was
used. Earlier implementation snapshots preserve their exact reported byte hashes.

GPU attempt01 failed its consecutive-forward equality check and remains
`FAIL_INCOMPLETE`. Bounded diagnosis confirmed unchanged inputs/model state but
advancing CPU RNG: pinned TAPIR inference calls `torch.randperm`. Different states
changed tracks by at most0.015114 raster pixels; restoring identical CPU/CUDA RNG
states makes every output bitwise equal. Attempt02 snapshots pre-forward RNG for
its repeat and restores the primary after-state for subsequent clips. No tolerance
or upstream model patch was used. All attempts and diagnostics remain separate.

Fresh-process checkpoint/RNG reload of the full first459-frame/55-query clip
matched tracks, uncertainty outputs,384-wide initial appearance,512-wide tokens
and time indices bitwise; after-RNG states also match. This is frozen inference
replay, not training resume, optimizer restoration or model selection.

## Scientific limits and next gate

The separate CPU scorer opens the hashed train archive after complete output
save. Initial assignment is maximum cardinality with bboxIoU>=0.5 AND own seed
inside GT, ascending GT/proposal IDs; no future metric chooses it. Future scoring
uses exact annotated frame IDs before the official cut, without interpolation,
reassignment or confidence filtering. This is an oracle association for diagnosis,
unavailable to deployment. The eight clips were previously selected/inspected,
so neither process isolation nor these scores establish blindness or generality.
Old ledger validation metadata was read; no validation feature array or sealed
audit was opened. Source cut=-1 on10212 remains unchanged.

The512-wide state has32 ordered xy/uncertainty samples and **initial-only**
appearance. It has no explicit token for a later-appearing object;10903's hidden
item annotation starts at224. Initial cover geometry does not prove preservation
of its identifying cue. Temporal identity and actor-relative left/depth semantics
also remain unresolved. Establish input-only prelude appearance/association before
a quality fit; do not extend the old failed head budget or provide GT IDs as inputs.
Decision/modality Accuracy/NLL/Brier/ECE are **NOT EVALUATED**. The complete
Omni/joint-input goal remains active and unmet.

## Execution, accounting and artifacts

[Fixed protocol](R2_MASK_TEMPORAL_PROTOCOL_20261014.md); code/tracker/reload source
`729afb994656260774b7261618213f03183d33d5`. Module SHA
`b21a7bf2dd1236a8ae3b8ec3248b68531508229ffed135206b421deddddc156e`.
[TAPIR source/checkpoint](R1_TAPIR_ADMISSION_20261013.md),
[SAM2](R1_SAM2_TINY_20261014.md) and [data rights](R2_PERCEPTION_TEST_SOURCE_20261012.md)
remain pinned; strict load/source hashes passed. Model-state before/after SHA
`3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a`.
RTX3080 Laptop16GB, Python3.11.9, Torch2.6.0+cu124, NumPy2.4.6, PyAV18.1.0;
FP32/TF32off, deterministic, seed17,4 CPU threads, chunk16; no upgrade.

Main body113.247705s; primary synchronized forward windows sum52.976868s.
Peak Torch allocation10,655,064,064bytes(~9.92GiB), reservation15,825,108,992bytes
(~14.74GiB). Imports/driver memory/inherited SAM2 cost are excluded. This is not
end-to-end Decision latency or exact total GPU time. Failed body13.560965s,
three-forward diagnostic11.640328s, fresh replay forward/compare4.164362s have
different scopes. CPU pooling0.567714s. CPU peak RAM unmeasured; cloud cost0.

TAPIR31,072,327 + SAM2 producer38,962,498 =70,034,825 registered component
parameters; primitives add0. This is not full Omni accounting. Their existing
FP32 weight files total280,256,538bytes before code/processors/readouts, exceeding
the provisional200MB package goal. No compression or encoder training occurred.

External R2 directories: `perception-mask-trajectory-smoke-01/`,
`perception-mask-query-plan-01/` and`-02/`, `perception-mask-query-trajectories-01/`
and`-02/`, `perception-mask-query-repeat-diagnostic-01/`, `perception-mask-query-reload-01/`.
Separate geometry scores and executed scripts/snapshots with hashes are in
the adjacent [publication JSON](R2_MASK_TEMPORAL_RESULTS_20261014.json).
Raw arrays/RNG states/snapshots remain local; no weights/media/NPZ enter Git.
Original teacher/model/feature artifacts were read only.

Main result SHA `94f22d64d6234d3626294c30b2c393845022b6811c0c0e7e08fdbb7c0ca9d178`;
config SHA `2a719e0c82f000b7354b282caf0e2aa2ee5e774e6d4c26846406498cca4758d5`.
Ruff passed;978 CPU tests passed and1 skipped in14.96s. Model/dataset manifests,
four catalogs and example policy audit passed. GPU/source diagnoses are outside
normal CPU CI.
