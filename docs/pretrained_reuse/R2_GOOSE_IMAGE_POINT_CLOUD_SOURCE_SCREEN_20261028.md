# GOOSE image and point-cloud source screen (2026-10-28)

Status: **candidate for a bounded real camera + sensor Path A probe; not yet
admitted for model selection or training.** No dataset content, sample labels,
media, checkpoint, or GPU was opened. This is a source and task screen only.

## Why this candidate

The official GOOSE site describes a real off-road robotics dataset with RGB/NIR
segmentation masks, pointwise-annotated point clouds, localization, multiple
platform/environment releases, and approximately 15,000 image/point-cloud
frames. The publisher explicitly states that the data is CC BY-SA 4.0. Its
official setup page gives separate public train, validation, and test archives
for both 2D images and 3D point clouds. The smaller validation archives are
listed as 2.9 GB (2D) and 3.3 GB (3D); full train archives are 22.5 GB and
27 GB, respectively. This appears more practically inspectable than sources
that require institutional access or have an unresolved dataset license.

Primary source pages, checked 2026-10-28:

- Project and license: <https://goose-dataset.de/>
- Dataset download sizes, release contents and license: <https://goose-dataset.de/docs/setup/>
- Platform descriptions: <https://goose-dataset.de/docs/platform-descriptions/>
- Class ontology: <https://goose-dataset.de/docs/class-definitions/>
- Dataset structure and example data-loading code: <https://goose-dataset.de/docs/examples/>
- Dataset paper: <https://arxiv.org/abs/2310.16788>

The source's explicit dataset license is stronger evidence than the Apache or
MIT license of unrelated source code. Preserve attribution and ShareAlike
obligations in any later corpus/package record. This is a research engineering
screen, not legal advice.

## Product relevance and limits

Camera plus point-cloud input fits a concrete embodied-decision setting:
choose among typed traversability or obstacle-response options from a robot's
current scene. The image and point cloud are synchronized spatial inputs, and
the publisher supplies labels for both modalities. However, the current
official task is semantic segmentation, not a demonstrated decision task that
requires both modalities. Either modality may solve many examples on its own.
We must not call the dataset evidence of multimodal complementarity without a
predeclared both-required task, matched image-only and point-cloud-only
baselines, and modality-removal / mismatch controls.

The official download page lists separate modality archives and reports test
assets without labels for some subsets. The source page does not establish
that image/point-cloud examples are joined by stable IDs across every archive,
that the validation/test split is grouped by recording sequence, or that no
same-scene near-duplicates cross splits. The dataset site describes additional
platforms in GOOSE-Ex; these must not be pooled without a platform- and
sequence-level split audit. The released test split must remain untouched for
final evaluation.

## Admission gates before any run

1. Pin the publisher pages, release version, archive URLs, file sizes and
   hashes; record any access redirects and the exact license text bundled with
   the data.
2. Check current free disk and transfer throughput before fetching anything.
   The official validation-only downloads total at least 6.2 GB before
   extraction; train archives are substantially larger. Do not start a transfer
   unless there is adequate space for the compressed and extracted files and
   the repository's disk reserve. A local read-only check on 2026-10-28 showed
   11.00 GiB free on C:, while the publisher does not provide extracted-size
   estimates on the inspected page. That is insufficient evidence to promise
   that both validation archives plus extraction and the reserve fit, so no
   download or three-sample throughput timing has been started.
3. Inspect only train/validation metadata first. Resolve image-to-point-cloud
   joins, timestamps, platform, sequence/recording IDs and exact split
   membership. Reject cross-split scene, source, media-identity and normalized
   content overlaps. Do not inspect test labels.
4. Audit candidate encoder pretraining overlap and preprocessing compatibility
   with both RGB/NIR and point-cloud inputs. Do not assume the existing visual
   checkpoint provides a useful point-cloud representation.
5. Freeze a compact decision task whose labels are valid, its class/option
   support is sufficient, and its target construction genuinely depends on
   both image appearance and 3D geometry. If no such task can be justified from
   the release's annotations, keep GOOSE as a paired unimodal-transfer source,
   not an M-2 both-required task.
6. Pre-register image-only, point-cloud-only, joint, and shuffled/mismatched
   controls with scene-grouped validation. Keep the test split sealed and report
   per-class and calibration metrics, not only pixel or sample accuracy.

## Source comparison

- **GOOSE:** best immediate follow-up candidate. The official dataset license
  is explicit and validation archives are individually listed; scene grouping
  and both-input task are still unknown.
- **OUTBACK:** the authors' README describes simulated RGB, LiDAR and IMU with
  train/validation/test, but the repository's Apache 2.0 statement does not
  itself establish the dataset/third-party asset license. Its SharePoint
  downloads could not be inspected by the read-only browser fetch. Keep at
  `REVIEW` pending explicit data rights and split details.
- **TartanDrive 2.0:** strong robot-sensor coverage, but its official data repo
  documents public downloads while the dataset's own reuse license and a
  group-safe evaluation split were not established in the checked publisher
  sources. Keep at `REVIEW`; the repository's MIT license is not a dataset
  license.
- **Zenseact Open Dataset:** official source states CC BY-SA 4.0 and broad
  camera/LiDAR/GNSS/IMU/control coverage, but downloading requires an access
  request; existing sequence annotations do not yet prove a both-input
  decision target. No access request was sent.
- **NVIDIA PhysicalAI AV:** the dataset card requires agreement to a separate
  NVIDIA dataset license and limits use to automotive-related use cases. Do
  not accept that agreement or use the source under the current scope.

## Work performed and next action

This screen used only official publisher pages and the dataset paper. No data
or model was downloaded, no user data or existing artifact was changed, no GPU
or cloud compute was used, and no license agreement was accepted.

Next: perform a metadata-only, hash-pinned GOOSE validation/train archive
preflight **only if** local disk can safely hold the selected subset and
extraction. First establish the required joins and sequence grouping from the
publisher metadata. If those checks fail or the task cannot be made
both-required without synthetic label construction, close this candidate
without model selection and continue source/task admission.
