# Training-only native grid correspondence support — fixed before execution

## Purpose and decision

The completed eight-scene masked-region pair fit overfits; only15 same-hidden
proxy pairs exist. Establish whether **all99 existing train scenes** offer useful
diverse hidden/cover correspondence supervision before another fit. This bounded
data-preparation extension is authorized by the user's autonomous/public-data
instruction. It preserves the full <=200M native Omni/95–98% goal.

No training, model inference, new downloads, validation arrays/annotations,
sealed audit, augmentation, capacity change or product promotion. The frozen
32-time×16×16×384 TAPIR context bank is referenced, not rebuilt or overwritten.
This is not a controlled data-scaling comparison to the eight-scene experiment:
grid inputs, label rule, role support and384 rather than768 channels differ.

## Fixed input and supervision contract

Require completed spatial corpus resultSHA
`a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da`,
bound configSHA`d17c1f1f9e55f17d58983603c1c8bf27ac171bed9d2ee04108535c84b15e35ff`,
and actual hash/shape/finiteness/native-time/grid checks for every train array.
Check99 unique media IDs and hashes against the protected train media manifest.
Do not open the existing validation arrays/annotation ZIP or any sealed set.
The prior report contains already-observed validation metadata; that is not a new
validation run. Confirm actual train assets' file hashes without decoding media.

Only after input verification, read ALLOW training annotations pinned at source
`google-deepmind/perception_test@3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`,
trainZIP SHA`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
Recheck each clip's task roles against the existing99-scene role auditSHA
`052f23f3282aafb3d7f4ed1a7775d6d9ad133954be465e6a1394898e7265d9b3`.
Use all grounded group objects plus hidden, cover a group subset. Other source
objects are outside this role namespace; never claim complete scene membership.

Grid extents are [x/16,y/16,(x+1)/16,(y+1)/16], row-major, fixed independently of
labels. At each original sampled frame, use range-limited source-box interpolation
with no extrapolation. A whole grid cell must be contained by exactly one available
role box AND have no positive-area overlap with another available role box.
Such a cell receives that source ID **for supervision only**. A contained cell with
competitors is ambiguous=-2; every other cell is unknown=-1. Retain all8,192 cells
per clip. Neither unknown nor ambiguous becomes a negative correspondence label.
This bbox rule cannot prove foreground membership: cells may include background;
missing source tracks can conceal competitors. `is_masked` flags are retained,
not interpreted as visibility. Annotation start is not physical object birth.

Enumerate all256×256 Cartesian pairs in the same56 forward windows:31 adjacent,
24 gap8,first→last. Both uniquely labelled endpoints are eligible; target1 iff IDs
match. Retain complete all/eligible/unknown/ambiguous/intersection ledgers, per
clip/stratum; save eligible pair indices/targets/strata and every native cell label.
Report source boxes/exact-interpolated-missing/flags, per-role native cell support,
same-hidden/same-group/hidden-group-negative/group-group-negative/cover subset,
distinct video/role instances, native-grid endpoints and time windows. No best
GT cell, target-based sampling, duplication, source ID as feature or easy-case drop.

This preparation creates no train/development split or selector and computes no
model quality. Future consumers must separate source supervision from input
features and group every pair from one media in one split. Do not train merely
because total pair count is large; report whether hidden support is spread across
scenes, and retain the sparse/zero-support scenes. A larger correlated pair count
does not repair missing semantic cue→cover memory supervision.

## Execution and integrity

New separate output directory, CPU/NumPy existing environment. Save effective
config, source commit, code/protocol/input hashes before label construction; rehash
protected inputs at exit. No legacy writes or process intervention. Body cap600s,
process peak4GiB, disk reserve8GiB, checked between stages/clips, not watchdog.
Keep incomplete attempts on faults; do not silently relax source/geometry gates.
Save/reload every output array exactly and hash files. Fresh whole-corpus CPU
repeat uses another directory and requires all per-clip arrays bitwise equal.
An independent scalar reference reconstructs every grid label from recorded source
box proxies, verifies pair eligibility/targets/role/window/exclusion counts and
saved inventory. This reference **does not independently reconstruct source
interpolation**; preserve that explicit limitation.

Tests first fail then cover row-major extents, whole-cell versus partial containment,
positive-overlap ambiguity, touching boundaries, absent source support, invalid
identity/geometry, role accounting, no input/RNG mutation and train-only identity
admission. Review before execution; Ruff/full CPU/manifests/exact publication-head
CI. Preserve completed rejected models and their scientific scope.
