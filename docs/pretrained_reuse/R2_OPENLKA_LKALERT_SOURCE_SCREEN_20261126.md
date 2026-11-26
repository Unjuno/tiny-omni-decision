# OpenLKA / LKAlert source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED`; metadata and paper review only.** OpenLKA/LKAlert is
the strongest newly reviewed task-fit lead so far for a bounded video-plus-
vehicle-sensor Decision task: predict whether a lane-keeping system needs a
warning, optionally return a short reason. It is not admitted for data access
or fitting because data-use rights are not explicit, public materials do not
establish a source-group-disjoint split, and the published label construction
may expose source/category shortcuts. No dataset payload, annotation rows,
sample videos, model weights, or evaluation data were accessed.

## Pinned identities and inspected materials

- Official project page observed 2026-11-26: <https://openlka.github.io/>.
- Official code repository `OpenLKA/OpenLKA`, `main` commit
  `493c7730f6a322b848a089eb44ecaf62c1b26d74` (GitHub API metadata and README
  only; no clone or media fetch).
- `OpenLKA` dataset paper: arXiv `2505.09092v2`.
- `LKAlert` task paper: arXiv `2505.11535v1`.
- No official dataset revision, versioned snapshot, or content hash was
  established. The official page says a versioned DOI will be archived later.

## Product and task fit

The proposed task is binary predictive alerting: from recent forward video
and synchronized CAN / vehicle-state signals, choose `alert` or `no_alert`
for an impending LKA failure, with an optional explanation. This is a concrete
Decision-style task with a plausible reason for both visual road cues and
vehicle dynamics to help. A calibrated option-probability JSON output could
be measured without requiring free-form generation.

The papers report 389.1 hours across 62 vehicle models. The current project
page reports 53 drivers and 322 unique route logs; it says 323 hours came from
four lab drivers and 66 hours from community contributors, with 83% recorded
in Tampa Bay. The full decoded per-route data, including 526×330 synchronized
video, is shared after review by request. The public GitHub repository has
sample folders, not a pinned full release or a verified train/validation
manifest.

## Data-quality risks that block a quality run

1. **Label/source shortcut.** The LKAlert paper states that its alert corpus is
   derived from failure and normal subsets; it labels every Failure sample
   `Yes` and every Normal sample `No`. A model may identify the source subset,
   recording conditions, or repeated temporal windows rather than learn a
   general warning rule. A future audit must inspect source-route identity,
   window proximity, category balance, and modality-only controls before
   fitting.
2. **Split independence is unverified.** The paper describes a held-out set of
   1,000 samples drawn from Failure and Normal and absent from training, but
   does not establish route-, driver-, vehicle-, or temporal-group separation
   in the accessible description. Adjacent windows from one route can be
   highly correlated. The source page describes repeated route passes and
   strong Tampa concentration. A row-level holdout is insufficient evidence
   of generalization.
3. **Annotation target is narrow and partly subjective.** The label asks
   whether an alert should be issued, but the paper's construction assigns
   class by Failure/Normal source. The event-triggering heuristics include
   lane-centering deviation and LKA disengagement, with a 3.5-second
   before / 2.5-second after window; failure labels also receive manual
   screening. The label's temporal boundary and false-alarm cost need a
   frozen, audited task contract.
4. **Published quality is modest and not comparable to the product target.**
   The paper reports 69.8% Accuracy and 58.6% F1 for Qwen2.5-VL-7B on its
   1,000-sample held-out set. This is research evidence for that task and
   protocol, not a small-model result, group-independent estimate, calibrated
   option-quality result, or evidence for 95–98% across modalities.
5. **Vehicle and geography coverage are clustered.** Route count and vehicle
   model count are not independent groups. The available source description
   does not provide enough identity detail to verify the roadmap's provisional
   minimum group floor or a cross-location split.

## Rights and access

The pinned GitHub README labels the **repository/code** MIT. The official
project page explicitly separates code/DBC MIT terms from the **data**, which
is governed by OpenLKA terms: cite and attribute, do not attempt
re-identification, do not redistribute raw video, and share derived tables
only with those terms attached. Full decoded data is provided after an access
review; raw logs require an inter-institutional research agreement. These
statements do not explicitly grant product/commercial training or derivative
model use. The arXiv paper's CC0 license applies to the paper, not the dataset.
Therefore data-use permission for this product goal remains unresolved. No
access request was sent, and no personal or organizational information was
submitted.

## Decision and next gate

Keep OpenLKA/LKAlert as a **high-priority, rights-pending data lead**, not an
admitted corpus. Before any download or label access:

1. Obtain explicit written dataset terms covering product-directed training,
   derived model retention, and permitted output/artifact distribution.
2. Obtain a pinned release manifest with driver, route, recorder, vehicle,
   timestamp/window IDs and label provenance; verify that a new train and
   validation split can be made group-disjoint without reusing exposed paper
   validation samples.
3. Audit whether failure/normal source membership or time-window adjacency
   predicts the label without meaningful visual or sensor evidence. Freeze a
   semantically reviewed target and compare video-only, CAN-only, and joint
   models plus source/time/metadata controls.
4. Only then consider a small frozen-feature Path A evaluation. BERT is not a
   remedy for source leakage; it may be compared later as a text component if
   a new approved task actually needs language.

This screen does not change any model, dataset, existing evaluation, or
checkpoint. No GPU, cloud compute, paid access, or external contact was used.

## Primary sources

- [OpenLKA official project page and data terms](https://openlka.github.io/)
- [Pinned OpenLKA code repository README](https://github.com/OpenLKA/OpenLKA/blob/493c7730f6a322b848a089eb44ecaf62c1b26d74/README.md)
- [OpenLKA dataset paper, arXiv:2505.09092v2](https://arxiv.org/html/2505.09092v2)
- [LKAlert task paper, arXiv:2505.11535v1](https://arxiv.org/html/2505.11535v1)
