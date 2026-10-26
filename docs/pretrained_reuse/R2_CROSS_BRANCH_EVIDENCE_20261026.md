# Cross-branch R2 evidence reconciliation (2026-10-26)

## Why this record exists

The repository has several concurrent draft PRs and remote research branches.
This read-only pass records relevant evidence so the active R2 plan does not
repeat an experiment or treat a source screen as a model result. No other branch,
PR, worktree, corpus, run, or checkpoint was changed.

## Direct evidence from open PRs and preserved runs

| Evidence | Result / gate | Consequence |
|---|---|---|
| [PR #21, CLEVRER synthetic AV coverage v2](https://github.com/Unjuno/tiny-omni-decision/pull/21) | On its task-specific development split, a frozen-encoder joint temporal LSTM scored 83.51% Accuracy / 0.3534 NLL / 0.2244 Brier / 0.0369 ECE at selected step 1,792; audio-only scored 75.14%; video-only and mean-pooled controls were at chance. Scene-paired bootstrap for joint minus audio Accuracy: +8.35 pp (95% interval +5.89 to +10.83). | Evidence of temporal AV interaction on CLEVRER plus a generated 880 Hz tone, not natural AV or general Decision quality. The evaluation is exposed development data with cross-task media and question-template overlap. Do not repeat or tune on these IDs. |
| [PR #27, synchronized AV source screen](https://github.com/Unjuno/tiny-omni-decision/pull/27) | KilometerAudio has no train split and its media rights are unresolved; EvaOmni has no established train/dev split and its referenced data-license file is missing; AVI-Bench prohibits model training under its data-use terms. | None of these sources is admitted for AV development fitting. The PR also records the completed CLEVRER synthetic run, so it is not a reason to repeat it. |
| [PR #17, MAVD/Urbansas source screen](https://github.com/Unjuno/tiny-omni-decision/pull/17) | Although listed as CC BY 4.0, train/validation MAVD annotations share parent-session prefixes; Urbansas locations are not shown disjoint from upstream test/unpublished content. The proposed class-level audiovisual co-occurrence label does not prove exact sound-to-object correspondence. | Review-only; no training, validation or audit admission. |
| [PR #19, AMI AV source screen](https://github.com/Unjuno/tiny-omni-decision/pull/19) | Metadata supports a recommended SA/SB participant split, but selected media windows are small/imbalanced, timestamp synchronization and visible-speaker labels remain unverified, and source-content overlap is incomplete. | Review-only; resolve synchronization and split/content identity before any model fitting. |
| [PR #30, AVSBench / NIST ASAPS source gates](https://github.com/Unjuno/tiny-omni-decision/pull/30) | AVSBench is CC BY-NC 4.0 with underlying YouTube media rights unverified, and its native output is masks rather than caller-option probabilities. NIST ASAPS requires a signed approved research-only data-use agreement; task labels, splits, group identities and commercial rights are unknown. | Neither source is admitted for product training or quality selection. |
| [PR #29, CommonsenseQA 2.0 text Decision probe](https://github.com/Unjuno/tiny-omni-decision/pull/29) | The frozen MiniLM plus shared 196,865-parameter readout selected 51.79% on dev, equal to majority-class accuracy, and predicted “no” for 93.5% of rows. | A second narrow text task failed to establish useful general text Decision transfer. Do not tune on its exposed dev IDs. |
| [QASC frozen-feature probe](R2_QASC_TEXT_DECISION_20261026.md) | Selected 21.81% Accuracy on an eight-choice QASC development set; best at epoch 1, with a strong train/dev divergence. | Text-only evidence remains task-specific and weak; no broader text claim or product promotion. |
| [iNaturalist CC0 frozen-image readout](https://github.com/Unjuno/tiny-omni-decision/blob/19392c947da150b4f0e4e1aca4cccef5374c1bdf/docs/pretrained_reuse/R2_INATURALIST_VIDEOPRISM_READOUT_20261026.md) | On 683 observer-disjoint development photos from a live-API CC0 freeze, frozen VideoPrism-B plus a 6,152-parameter readout selected 81.13% macro Accuracy / 0.5584 macro NLL. Zero-image Accuracy was 12.59%; permuted-image Accuracy was 10.69%. The 2,048-epoch curve selected epoch 1,276; train Accuracy kept rising while validation NLL degraded afterward. No sealed audit exists. | Useful natural-image dependence evidence for a closed eight-species task, not arbitrary option-conditioned image decisions. Treat live-source/community labels, exposed validation and train-side visual spot-check concerns as limitations; keep data and run external to Git. Latest source-branch commit `19392c9` passed its public test workflow on this recheck; it is not in the open-PR list. |
| [Synthetic AV architecture follow-ups](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-temporal-av-alignment-20261026/docs/pretrained_reuse/R2_SYNTHETIC_AV_RESIDUAL_GATE_RESULTS_20261026.md) | On the already exposed generated-tone/CLEVRER task, query-conditioned and modality-bypass gates stayed at chance through 16,384 updates. An identity-initialized residual gate matched the raw-input LSTM at 100% validation Accuracy; aligned AV scored 100%, while audio-only, video-only, missing and mispaired controls fell to chance. | The raw recurrent path solves this generated rule; the residual gate adds no demonstrated gain over the LSTM. Synthetic mechanics only; do not repeat or tune these IDs or infer natural AV quality. |

The PRs above are open drafts and are cited as independent concurrent evidence,
not merged into this branch. Their source commit identities are visible in the
GitHub PR records. Their independent reports and external artifacts remain the
source of detailed hashes, protocols and metrics.

## Current decision

The synthetic AV result is a bounded feasibility result. The latest source gates
do not provide a rights-cleared, adequately grouped natural-AV task with a
development split where both streams are demonstrably required. Therefore the
next justified R2 action is further source/task admission work, not another fit
on exposed CLEVRER data or on challenge test splits. Any admitted source must
have verified source/media rights, parent-asset grouping, temporal alignment,
and labels that require each claimed input; matched unimodal and intervention
controls must be fixed before training.

Additional independent-branch evidence since the initial reconciliation:
iNaturalist CC0 yields an 81.13% macro-Accuracy frozen image readout with
strong zero/permuted-image controls, but train/validation divergence after epoch
1,276 and unresolved label-grounding concerns make data/task diagnosis the next
image step. The recent synthetic AV residual-gate comparison reaches the same
100% exposed validation Accuracy as the simpler recurrent baseline, so it does
not justify adopting another gate. Neither result closes natural AV or the
four-modality product gate.

No evidence here closes the four-modality quality requirement, total <=200M
resident-parameter gate, <=200 MB package aspiration, or R7/R8 device/release
gates. R0 deployment device and task-specific confidence/time/cost criteria also
remain open. The source screens do not establish an encoder-family limit.
