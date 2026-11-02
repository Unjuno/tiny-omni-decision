# Hoi! source and task screen — 2026-11-02

## Disposition

**`REVIEW_NOT_ADMITTED` for current R2-M product-quality evaluation.** Hoi!
is a strong conceptual fit for force-grounded interaction because it contains
synchronized video, force/torque, and tactile measurements. Its published
challenge, however, uses RGB to predict force targets; it does not establish a
decision task that requires video and force as simultaneous inference inputs.
The challenge evaluation is only 40 interactions across two kitchens, while
the official dataset page reports 38 environments. No group-disjoint
train/validation/audit split for this project has been verified. The full
deposited release is far beyond local storage capacity, though selective
download feasibility remains unknown. The README and one tiny
interaction-window annotation were retrieved; no media, sensors, or challenge
labels were opened. The annotation is source-structure evidence only and was
not used for training or evaluation.

## Official source evidence

- The [official dataset page](https://hoi-dataset.ethz.ch/) reports 3,048
  sequences, 381 articulated objects, 38 environments, and four embodiments.
  It describes aligned RGB views, force/torque, and DIGIT tactile data. It
  states CC BY 4.0 and says commercial use is allowed with attribution.
- The [official challenge page](https://hoi-dataset.ethz.ch/workshop.html)
  defines T1 as RGB-based articulation type/axis estimation and T2 as RGB-based
  prediction of peak force and mechanical work. T2's force sensor is the source
  of the target; it is not an inference-time input. The challenge uses 40
  interactions from `kitchen_9` and `kitchen_16`, with labels hidden on
  Codabench.
- The [deposited README](https://bonndata.uni-bonn.de/api/access/datafile/54498)
  (persistent ID `doi:10.60507/FK2/QODWTV/1ELBAW`, 11,277 bytes, SHA-256
  `973b02605c7bda0fbe965e3f6ece9b24ebfd7f2263fb63a8a8ab552eac88fac0`)
  states `CC-BY`, describes EgoBlur anonymization, and documents files named
  `interaction_splitting_info_*`. It does not specify participant consent
  scope.
- One public [confirmed interaction-window annotation](https://bonndata.uni-bonn.de/file.xhtml?persistentId=doi%3A10.60507%2FFK2%2FQODWTV%2FKBKXAY&version=1.0)
  (persistent ID `doi:10.60507/FK2/QODWTV/KBKXAY`, 664 bytes, MD5
  `d9cf961d71420e7fb44ebda6831e334d`, SHA-256
  `f46ec180a6f3fd9d4f0ec459e4d7676eff1fabd6fe40177737dcd05ca077f3c4`)
  contains four timestamped `open`/`close` windows for one interaction. This is
  session-level event segmentation, not a train/validation/test split; one
  example cannot establish group independence.
- The public metadata page for one
  [`camera_rgb.zip`](https://bonndata.uni-bonn.de/file.xhtml?persistentId=doi%3A10.60507%2FFK2%2FQODWTV%2FEUSQRS&version=1.0)
  reports 1.4 GB for a `bathroom_1/gripper/aria_gripper` package associated
  with interaction indices `1-2` (MD5 `d920385292d0803a43f33afe3137ee5e`).
  This is one example archive, not a measured total for a candidate subset;
  its ZIP and any sensor archive were not downloaded.
- The [Bonn Dataverse record](https://bonndata.uni-bonn.de/catalog/datasets/10.60507/FK2/QODWTV/)
  metadata inventory reported 4,769 files totaling 3,186,202,640,224 bytes
  (about 2,967.38 GiB), including 3,181 ZIPs totaling 3,185,687,461,273 bytes
  (about 2,966.90 GiB) and 1,354 JSON files totaling 156,171,537 bytes. The
  site allows selective file download, but whether an aligned video+sensor
  subset fits local resources has not been assessed; one public two-interaction
  RGB archive is already reported as 1.4 GB. No archive contents or
  annotation files other than the single event JSON above were accessed. The
  README and dataset page state CC BY; participant
  consent scope remains unknown.
- A complete metadata-only inventory for Dataverse version 1 was retrieved as
  a 3,769,110-byte API response (SHA-256
  `a7aa8d7f2bb362812ea622c37aeac4192999ce555e91c7679ec555ae5c81a172`). It
  lists 22 force/torque ZIPs (136,626,600 bytes), 22 DIGIT ZIPs
  (2,858,497,449 bytes), 24 UMI telemetry ZIPs (23,595,019 bytes), and 24
  odometry ZIPs (139,860,733,262 bytes). The 362 `camera_rgb.zip` files total
  855,857,508,336 bytes. Matching the 24 UMI telemetry packages to the
  `aria_human` RGB view involves 46,809,692,711 bytes of RGB archives plus
  23,595,019 bytes of telemetry if acquired in full; none of these archives
  were downloaded.
- Parsing interaction-index suffixes from force/telemetry package paths gives
  168 force-side keys and 187 telemetry-side keys, with a deduplicated union
  of 198 `(environment, index)` keys. This is a metadata-derived candidate
  count only: the keys have not been verified against all event annotations or
  physical-object identities, so independent group count remains UNKNOWN. The
  candidate is below the roadmap's provisional 200-scene/clip floor even
  before allocating train/validation/audit splits.
- The [official processing tools](https://github.com/timengelbracht/hoi-dataset-tools)
  describe data processing and alignment utilities; their existence does not
  resolve the task/split/resource gates.

## R2 decision

Hoi! is not admitted for the current both-input-required Decision evaluation.
The two-kitchen challenge split is too small to support the roadmap's
provisional 200 independent-scene/clip floor, and it is a hidden-label external
challenge rather than a project-controlled split suitable for the needed
matched unimodal controls. The full dataset reports 38 environments and 381
objects. Object-disjoint grouping might provide more independent groups than
environment grouping, but object identity and cross-embodiment/session overlap
have not been audited, so the available group count is UNKNOWN. The full release
size exceeds local C: free space by over two orders of magnitude. The matched
UMI video+telemetry archives total about 43.6 GiB; they could be processed
serially, but the metadata-derived group candidate count remains below 200.

Do not ingest Hoi! for R2-M under the current gate: the metadata-derived
sensor-bearing group candidate count is below 200, and the full selected UMI
video+telemetry sweep spans about 43.6 GiB. Reopen only if a valid object
identity mapping establishes at least 200 independent groups, a smaller legal
feature-extraction route is demonstrated, and a task is frozen that needs both
inference streams. No media/sensor payload, challenge labels, sealed audit,
GPU, fit, or evaluation was used. The README and one event-annotation hashes
are above.

Machine-readable record: `manifests/pretrained_reuse/hoi-source-screen-20261102.json`.
