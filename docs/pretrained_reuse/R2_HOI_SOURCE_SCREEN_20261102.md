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
deposited release is far beyond local storage capacity. No payload was
downloaded or opened.

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
- The [Bonn Dataverse record](https://bonndata.uni-bonn.de/catalog/datasets/10.60507/FK2/QODWTV/)
  metadata inventory reported 4,769 files totaling 3,186,202,640,224 bytes
  (about 2,967.38 GiB), including 3,181 ZIPs totaling 3,185,687,461,273 bytes
  (about 2,966.90 GiB) and 1,354 JSON files totaling 156,171,537 bytes. This
  was an API metadata-only inventory; no archive contents were accessed. The deposited
  documentation and participant-specific downstream consent scope have not
  been reviewed, so practical product-use rights remain unverified despite
  the public CC BY statement.
- The [official processing tools](https://github.com/timengelbracht/hoi-dataset-tools)
  describe data processing and alignment utilities; their existence does not
  resolve the task/split/resource gates.

## R2 decision

Hoi! is not admitted for the current both-input-required Decision evaluation.
The two-kitchen challenge split is too small to support the roadmap's
provisional 200 independent-scene/clip floor, and it is a hidden-label external
challenge rather than a project-controlled split suitable for the needed
matched unimodal controls. The 38 reported environments also do not by
themselves provide a verified split or enough independent groups. The full
release size exceeds the local C: free space by over two orders of magnitude.

Do not download the full release or use challenge labels. Reconsider only if
the authors provide a legally usable, selectively downloadable subset with a
predefined environment/object-disjoint split and a decision target whose
inference-time label demonstrably depends on both video and force/tactile
input. No sealed audit, GPU, fit, or evaluation was used in this screen.

Machine-readable record: `manifests/pretrained_reuse/hoi-source-screen-20261102.json`.
