# MEVA video + GPS source/task screen — 2026-11-16

## Disposition

**`REVIEW_NOT_ADMITTED` for a Decision-quality run.** MEVA is a substantially
better rights and synchronization candidate than the YouTube-derived AV
benchmarks screened so far: the official release says CC BY 4.0, its paper
reports IRB oversight and signed participant consent, and its public GPS
metadata describes synchronized logger tracks. However, the currently visible
annotation split is not independent at the day/session level, and no
label-grounded task has yet demonstrated that both video and GPS are necessary.
Do not download media/GPS payloads or train from this screen.

## Pinned source evidence

- Official MEVA website, accessed 2026-11-16: <https://mevadata.org/>. The page
  says all MEVA data is CC BY 4.0, lists public ground-camera and UAV releases,
  and points to the annotation/GPS repository. Its website content is not
  versioned; this access date is the pin.
- Official data/annotation repository:
  <https://gitlab.kitware.com/meva/meva-data-repo>, `master` commit
  `421841a75577b697c314e952e585aecbb1b99e17` (2021-12-13). Its root LICENSE
  names the MEVA dataset and CC BY 4.0. Its README distinguishes
  `kitware-meva-training` annotations from `kitware` evaluation-level
  annotations.
- Official GPS README at the pinned repository commit says 116 GPX files cover
  released-video five-minute time slots; GPS timestamps are UTC while video
  filenames use local MUTC (UTC-4); logger IDs are stable across March
  collection days but not between March and May. It says the GPS payload is
  released under the MEVA license. The GPS ZIP itself was not fetched or opened.
- The WACV 2021 paper reports approximately 100 actors, 37 activity types,
  roughly 2.7 million GPS trackpoints from 108 loggers, and states that all
  actors signed consent forms after an IRB-reviewed collection. It also says
  actor numbers were matched to GPS logger IDs and enrollment photos. See
  <https://openaccess.thecvf.com/content/WACV2021/papers/Corona_MEVA_A_Large-Scale_Multiview_Multimodal_Video_Dataset_for_Activity_Detection_WACV_2021_paper.pdf>.
- The official AWS Open Data Registry lists the public S3 bucket
  `mevadata-public-01` and says access is no-cost. The official README lists
  about 470 GB for the KF1 video release; no media or S3 payload was accessed.
  See <https://registry.opendata.aws/mevadata/> and
  <https://mevadata.org/resources/README-meva-kf1-data.html>.

## Metadata-only inventory and integrity findings

The pinned Git repository was queried with a blob-filtered, no-checkout clone.
Only the top-level README, the MEVA annotation README, the GPS README, the
repository license, and one training-level annotation example were read. No
evaluation-level annotation contents, video, image, enrollment photo, GPS
track, model, feature, or sealed data were read.

Counting annotation filenames without opening evaluation annotations found:

| Annotation tier | Unique clip IDs | Collection dates |
|---|---:|---:|
| `kitware-meva-training` | 1,443 | 7 |
| `kitware` evaluation-level | 769 | 7 |

The exact clip-ID intersection is zero, but **all seven dates occur in both
tiers** (`2018-03-05`, `03-07`, `03-09`, `03-11`, `03-12`, `03-13`, and
`03-15`). The public metadata therefore does not establish independent
day/session, actor, scenario-take, or synchronized cross-camera groups. The
paper says logger IDs persist across March days, so a clip-level split would
also allow repeated actors and shared five-minute scenes across tiers. The
source has 108 logger identities in the March release; actor-disjoint
validation would need a defensible mapping from annotation track IDs to those
logger IDs and appears below the roadmap's provisional 200 independent
scene/clip validation floor if logger identity is the group.

One training-level annotation example contains activity labels and local
`id1`/`id2` track references. The public material inspected so far does not
prove a direct, machine-readable mapping from those per-clip references to GPS
logger IDs. The paper's actor-number/photograph/logger association alone is not
enough to assert that this join is available in the released task files.

## Decision task and gates

A plausible future task is a bounded actor-and-event decision that joins a
video observation with synchronized GPS evidence. That is only a hypothesis:
the released annotations are video activity labels, not supplied-option
video+GPS decisions. GPS may be merely redundant or may leak scene/time priors.
Before admission, a frozen task must define candidate options and prove with
matched controls that both streams are needed: video-only, GPS-only,
time/location-prior-only, shuffled GPS, missing GPS, and candidate-order
controls. It must also establish participant/scene/take-disjoint train and
validation groups, resolve timestamp conversion, and preserve the public
evaluation tier for one preselected final evaluation.

| Gate | Finding | Disposition |
|---|---|---|
| Product-compatible data rights | Official release says CC BY 4.0; paper reports IRB and signed actor consent | Promising; cite exact source and preserve attribution |
| Synchronized sensor input | GPS README describes timestamped five-minute GPX slots and logger IDs | Structurally promising; no GPS payload inspected |
| Supplied-option Decision labels | Existing labels are activity annotations, not a supplied-option Decision task | Not established |
| Both-input necessity | No matched video/GPS controls exist | Not established |
| Independent groups | 1,443 train / 769 evaluation clips; all seven dates overlap, actor/session grouping unresolved | Fail closed |
| Exposure | No project MEVA corpus, media, GPS, or evaluation labels were opened in this screen | No clean-data claim beyond this bounded audit |
| Local feasibility | Full published release is hundreds of GB; a lawful, representative subset and measured extraction plan are not yet frozen | Unknown; do not bulk download |

Do not use NIST sequestered ActEV material. Reconsider MEVA only after the
source owner/metadata supports a strictly grouped, sufficiently sized split and
a task design with measurable both-input necessity. This screen does not
authorize model fitting or quality claims.

## Reproduction record

- Repository source commit: `421841a75577b697c314e952e585aecbb1b99e17`.
- Source inventory command: `git ls-tree -r --name-only HEAD`; filename counts
  were computed after canonicalizing `.activities.yml`, `.geom.yml`,
  `.types.yml`, and `.json` suffixes to clip IDs.
- Local metadata-only clone: `C:\CodexArtifacts\pretrained-reuse-r2\meva-source-screen-20261116\upstream-metadata`.
- No model/GPU work, media retrieval, or evaluation was performed.
