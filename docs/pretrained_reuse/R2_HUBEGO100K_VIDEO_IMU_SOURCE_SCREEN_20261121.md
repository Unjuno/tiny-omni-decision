# HubEgo-100K video + IMU source screen (2026-11-21)

## Status

**Promising candidate; not admitted and no payload was opened.** This is a
metadata-only source screen for a possible fresh R2-M video + sensor task. The
official Hugging Face repository is access-gated; the login/access form was not
submitted. No account, name, organization, country or intended-use details
were provided. The dataset, its action labels and its contributor catalog were
not downloaded or read.

## Why it merits the next gate

The dataset card and paper describe v1.0 as 72,754 episodes, 10,124 hours of
1080p first-person video paired with phone inertial streams and episode-level
action and language annotations. The paper reports 12,227 contributors,
15,548 contributor-specific environments, 35 action families and 56 task
types. The release declares CC BY 4.0 for commercial use with attribution and
states the recordings are first-party with no third-party rights restrictions.
The card and paper both require a short access form before content access.

The paper recommends contributor-disjoint evaluation and says a
task-stratified split is available on request. Its public datasheet says the
episode catalog includes stable `contributor_id` and `recording_id` keys,
making contributor and source-recording grouping feasible in principle.
Unlike the recently measured two-participant audio/vibration task, this source
has enough reported independent contributors to support a much stronger
generalization design.

## Immutable public metadata snapshot

- Hugging Face dataset: `HubLabs/HubEgo-100K`.
- Repository revision observed: `fe33edfcc9bafea12b85d5d81544b7b121ac83d5`.
- API `lastModified`: `2026-11-19T22:11:40Z`; API marks repository `gated: auto`.
- License declared in the card: `cc-by-4.0`.
- The public recursive file-tree endpoint returned 1,000 path records, of
  which 787 were files. It exposed 778 Parquet paths below `mono/data`,
  totaling 24,708,650,753 bytes, plus public metadata/card files. This is only
  the visible file listing, not the size of the complete media release; the
  access-gated video and MCAP payload was not inventoried or fetched.
- Canonical SHA-256 of the sorted `type|path|size|oid` rows from that API tree:
  `372acce28e7b1099876bdf2258118e3143fb003b3ae63de08e7426baecf117ba`.
- The source metadata summary is recorded in
  [`hubego100k-source-screen-20261121.json`](../../manifests/pretrained_reuse/hubego100k-source-screen-20261121.json).

## Proposed bounded Decision task after access

Use an episode-local four-choice action-family Decision: from a fixed temporal
video window and the time-aligned phone IMU window, select the action family
among four typed action IDs. The action-family labels are machine-generated;
the paper explicitly says they have not been audited by human reviewers.
Treat label noise as a first-order risk and do not call these labels verified
ground truth.

Before fitting, an access-approved follow-up must inspect only the permitted
non-audit catalog/annotations and freeze a contributor-grouped split. Every
episode from one contributor and every window from a source recording stays
in one partition. Choose eligible action families using a predeclared
train-only group-support rule; if there are too few common classes across
partitions, stop before fitting. Do not construct windows across episode
boundaries.

The first comparison should use the existing frozen VideoPrism visual path,
the timestamp-aware sensor stem, and the existing lightweight gated fusion
readout. Compare video-only, IMU-only and paired inputs with the same rows,
options, parameter accounting and training budget. Add shuffled-IMU,
within-recording time-offset, missing-stream and option-order controls. A
metadata/class-prior baseline must be fixed before fitting. Select checkpoints
on participant-grouped validation only. Report participant-level results and
calibration, not overlapping windows as independent samples. The pair is not
“both-input-required” until the predeclared controls establish a reliable
paired-input gain on held-out contributors.

This is a proposed task, not a completed split or an admitted benchmark. In
particular, the label generator is unpublished, the labels lack human review,
the access form remains outstanding, and the public card does not provide the
actual split file. A model result could not be used to claim broad video,
sensor or Omni quality.

## Decision and next gate

Do not access or acquire the data until the user has completed the official
access form or otherwise supplied an access-approved local copy and its terms.
After access, first verify repository revision, file hashes, actual media size,
catalog schema, contributor/recording identity, annotation provenance,
video/IMU alignment and split feasibility. Only then freeze the exact
four-choice construction and train/validation/sealed-audit group assignments.
Do not use the test/audit split during development. If the form, schema, or
group support makes the task unsuitable, retain this as a screened-out source
and continue to another rights-cleared candidate.

## Primary sources

- [Official Hugging Face dataset card](https://huggingface.co/datasets/HubLabs/HubEgo-100K)
- [Dataset paper and datasheet (arXiv:2611.21938)](https://arxiv.org/abs/2611.21938)
- [Creative Commons Attribution 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode)
