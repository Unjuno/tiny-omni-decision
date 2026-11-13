# R2: source-role supervision coverage — measured 2026-10-14

Following the [per-time appearance probe](R2_TEMPORAL_APPEARANCE_20261014.md),
the pinned training annotation archive was actually checked for all99 existing
occlusion train pairs,198 questions. No validation/test/sealed data, media decode,
model inference or training occurred. Existing source/config/corpus artifacts
were read only. This checks candidate supervision consistency, not quality.

All99 clips have consistent grounded hidden-item, covering-object and similar-group
labels. Track references resolve; hidden/cover are distinct singletons, cover is
in the group, `is_occluder` agrees, and the upstream count answer matches group
size. These annotations may supervise training losses; they must not choose
inference inputs, GT crops, masks, frame cuts or model predictions.

| Measured training coverage | Count |
|---|---:|
| Source-consistent clips | 99/99 |
| Original questions/correct option texts match | 198/198 |
| Groups of2/3/4 | 18/78/3 |
| Hidden track first annotated after frame0 | 13 |
| No source shuffle action annotated | 13 |
| Initial/final correct option differs | 53/99 pairs |

Missing shuffle annotation does not mean missing motion. First annotation time
does not establish true visibility. Actor viewpoint, game-time detection and
input-only hidden-target binding remain unresolved. Source supervision alone
does not prove that a trained model can use the inputs or that the dataset is
large enough. Do not remove the two/four-object groups to simplify the task.

## Corrected audit attempt, preserved evidence

Attempt01 compared upstream option order and integer answer IDs literally with
the existing canonical sorted options. This incorrectly flagged85 initial and81
final questions. Those mismatch counts are **invalid**, not corpus errors.
Source `occlusion_pairs.build_pairs` deliberately sorts option strings and remaps
the correct string to its new index. Attempt02 compares sorted sets and correct
option text; rebuilding all99 pairs from the source gives exact equality.
Existing answer-remapping regression passed:1 test in0.05s.
No corpus, production code, label or target was changed. Both attempts retained.

Executed source commit `551a70ad3035d44930c5583e41ea78cd7fce9b53`.
Source revision `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`;
training ZIP SHA `9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`;
unchanged train-pair SHA `8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7`.
Corrected CPU audit body4.374923s; GPU time0, cloud cost0, no environment upgrade.
Full99-row results and executed scripts remain at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-role-supervision-audit-02/`
and its parent; failed comparison attempt01 preserved separately.
The [small ledger](R2_ROLE_SUPERVISION_20261014.json) pins both attempts and runners.

## Consequence

Supervision exists for a predeclared input-only association experiment. It does
not justify feeding source track IDs into a Decision readout or treating them as
predictions. Specify deployable target/cover outputs, background/no-match handling,
fixed controls and validation protocol before fitting. Preserve all original
options and semantic remapping. No new quality result or model adoption here;
four-modality/OOD/fusion/package requirements remain unmet.

Publication verification: Ruff passed;978 CPU tests passed,1 skipped in14.81s.
Manifests/catalogs/policy audit passed earlier in the same unchanged-code turn;
log `C:/CodexArtifacts/pretrained-reuse-r2/temporal-appearance-manifest-validation-01.log`.
