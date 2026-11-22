# R2 task-admission reconciliation — 2026-11-14

## Scope

This dated addendum reconciles the 2026-11-04 R2 admission matrix with the
newer AMI, CATER, TartanAviation, and Localized Narratives evidence available
on 2026-11-14. It does not rewrite earlier experiment records, admit a source,
or authorize another training or evaluation run. No data payload, sealed audit,
or model was opened for this reconciliation.

Roadmap base: `docs/pretrained-reuse-roadmap-20261010` at
`bfb13bf6e9ac4f2ce244eb51e470afa907587380`.

## Updated candidate disposition

| Candidate | Latest evidence | Admission status | Next allowed action |
|---|---|---|---|
| AMI active-speaker AV | The existing preflight already inspected train/development annotations and media. Its 111 reviewed development candidates include 42 with no clear target face (34 with no discernible person); target-speaker truth, useful low-bitrate visual cues, and AV alignment remain unverified. The task cohort used four SA and five SB parent sets, too few independent groups for the project's robust task gate. ES2014 evaluation annotation was previously read in memory and its full group is quarantined; do not access SC unseen evaluation data. See the [preflight at `6d722c2`](https://github.com/Unjuno/tiny-omni-decision/blob/6d722c258f188a0a09c624bf7b39e3f352d55319/docs/pretrained_reuse/R2_AMI_MEDIA_PREFLIGHT_20261029.md). | `REVIEW_NOT_ADMITTED`; do not repeat or expand this cohort. | None on existing AMI groups. A separately sourced, non-exposed cohort would require a new admission review and cannot use SC. |
| CATER temporal video | The recovered fixed-validation run has 1,120 queries over 308 video groups. Video+question Accuracy was 18.86% versus 19.50% question-only; static, shuffled, reversed, and swapped-video controls were effectively unchanged. This validation was used for selection. See the [recovery report at `c23c8c5`](https://github.com/Unjuno/tiny-omni-decision/blob/c23c8c5ffb6fe8736c79a58b562bb1defbe8b83a/docs/pretrained_reuse/R2_CATER_TEMPORAL_READOUT_RECOVERY_20261114.md). | Local development source check passed for that fixed probe; it is not an admitted M-2 task and gives no evidence of video dependence. | Preserve as a negative task-specific result. Do not retune on these validation IDs. |
| TartanAviation and its two metadata derivatives | Source/task review found labels insufficient and partly circular, independent group assignment unresolved, and both-input necessity unproven. The exposure reconciliation registers 40,899 known parent-clip identities per revision while marking each history `INCOMPLETE`; all proposed groups fail closed. See [the ledger reconciliation in PR #100](https://github.com/Unjuno/tiny-omni-decision/pull/100). | `REVIEW_NOT_ADMITTED`; known exposure is not a clean-cohort certificate. | Do not use or acquire these revisions unless complete source history, valid independent labels, and a both-input task are established. |
| Localized Narratives / COCO | Metadata-only review found all 100 official validation annotators also in train shard 0; Flickr image-owner attribution and voice-recording rights/consent are unresolved. No media or model was accessed. See the [source screen at `6d722c2`](https://github.com/Unjuno/tiny-omni-decision/blob/6d722c258f188a0a09c624bf7b39e3f352d55319/docs/pretrained_reuse/R2_LOCALIZED_NARRATIVES_SOURCE_SCREEN_20261113.md). | `REVIEW_NOT_ADMITTED`. | No media acquisition or fitting. Reconsider only with authoritative image and voice rights plus a fresh annotator- and image-disjoint split. |
| SIMMC 2.1 situated shopping dialogue | The upstream repo is pinned at `6e3de4e3d11e2914edc3ab9123e6b0eb88933237`. Its `LICENSE` identifies CC BY-NC-SA 4.0, and its data README says the train/dev/test-dev/test-std divisions are random dialog splits (7,307 / 563 / 1,687 / 1,687). It describes text/dialog context plus scene-image grounding tasks, but does not establish scene-disjoint groups or that both modalities are necessary. Sources: [pinned LICENSE](https://github.com/facebookresearch/simmc2/blob/6e3de4e3d11e2914edc3ab9123e6b0eb88933237/LICENSE) and [pinned data README](https://github.com/facebookresearch/simmc2/blob/6e3de4e3d11e2914edc3ab9123e6b0eb88933237/data/README.md). | `REVIEW_NOT_ADMITTED` for this product-directed track; NC-SA terms fail the current rights gate, and split grouping is not established. | No data/LFS/media access. Reconsider only after rights scope is approved and a source-level scene/asset-disjoint task protocol is established. |

The earlier matrix remains the record for RoboMNIST, STARSS23, Eidon, and
PhysicalAI candidates; their existing negative or unresolved dispositions
remain in force. See
[`R2_TASK_ADMISSION_MATRIX_20261104.md`](R2_TASK_ADMISSION_MATRIX_20261104.md).

## Gate conclusion

No currently reviewed candidate passes all R2-M gates at once:

1. rights and source revision are sufficiently established for the intended
   use;
2. the task has valid, non-circular labels and a defensible group split;
3. the candidate cohort is not already exposed or quarantined;
4. a frozen, predeclared experiment can compare matched unimodal, joint, and
   input-destruction controls; and
5. the number of independent groups supports the planned uncertainty analysis.

Therefore, no new fusion fit, checkpoint selection, or R2-M quality claim is
justified by these records. Do not reopen exposed validation sets or sealed
evaluation groups to force a candidate through admission. Continue only after a
fresh source passes those gates; otherwise work on a separately admitted,
non-exposed Path A modality-native task under the existing R0/R2 protocol.

The first fresh candidate screened after this reconciliation, SIMMC 2.1, was
closed at the metadata/license gate without fetching data. This is a source
disposition, not a model-quality result.

## State and verification

- This is a documentation-only reconciliation on a branch based on the current
  PR #15 roadmap head.
- No existing worktree, PR, dataset, cache, model, checkpoint, or running
  process was changed.
- No model training, GPU evaluation, dataset download, or sealed-audit access
  was performed.
- This record does not close R2-M, R2 as a whole, or later R3–R8 gates.
