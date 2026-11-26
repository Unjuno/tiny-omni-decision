# Perception Test source-candidate task disposition — 2026-10-29

## Decision

Do not spend further validation-guided tuning on the current Perception Test
audio/video source-candidate readout. The task may remain a train-source
diagnostic, but the saved labels and interventions do not establish that it is
a reliable, genuinely joint audio/video Decision task. This is not a claim that
Perception Test or its pretrained representations are generally unsuitable.

This report reconciles existing external artifacts only. No media, labels,
official test data, sealed audit, model, or checkpoint was newly loaded. No
training or validation evaluation was run for this report, and none of the
artifact directories was changed.

## Frozen artifact provenance

Artifacts are retained outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2/`:

| Artifact | SHA-256 |
| --- | --- |
| Annotation-semantics result JSON (`perception-source-annotation-semantics-20261027-attempt-01/result.json`) | `6f599092a121dbbf186dcd646a5b57c3ad234084efad905d3b75a1e4a87d96e1` |
| Fixed train-source OOF result JSON (`perception-source-balanced-set-summary-20261027-attempt-01/result.json`) | `33bbe0995f17f856937e1e1846952128526214c70ff8ea607974021d19121620` |
| Balanced-set comparison result JSON (`perception-source-balanced-set-comparison-20261027-attempt-01/result.json`) | `6dec046ab35125acbac61d5c24348c857d75178c3c74d31ec1b2a20ae6422b92` |
| Same-label audio intervention result JSON (`perception-source-same-label-interventions-20261027-attempt-01/result.json`) | `30471d27f3c9dff4ba6427dbc5fe734bda97927a5b55ec554733252e64ed8b24` |
| Stratified annotation-review sample result JSON (`perception-source-semantics-review-sample-20261027-attempt-01/result.json`) | `9d7df2419a4b3b63b23d1da94f571cf2ba29f3492b433caad0d530c756531126` |

The reconstructed labels SHA-256 is
`c5a8396a6b47a8635033e445c67233c76a6ecb17d27f4e6943a153fd3598600d`; the
annotation archive SHA-256 recorded by the reconstruction is
`9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`.
The OOF run reports seed 17, three video-group folds, seven epochs, CPU
execution, and `sealed_audit_opened: false`.

## Findings

The reconstructed source task has 2,593 rows from 320 video groups. Of 692
`none` labels, 647 have no parent-object annotation; 47 have a known parent,
and only 30 have a known parent among the candidate set. Among 1,901 visible
records, 1,102 have multiple annotated parents. These counts make a simple
positive-versus-`none` target materially ambiguous without a more defensible
label protocol.

The saved three-fold train-source out-of-fold exact-set scores were:

| Features | Exact-set Accuracy | Macro record-candidate NLL | Macro record-candidate Brier |
| --- | ---: | ---: | ---: |
| Audio | 7.79% | 0.5191 | 0.1680 |
| Video | 16.81% | 0.3531 | 0.1176 |
| Joint | 18.63% | 0.3664 | 0.1212 |

The paired joint-minus-video OOF Accuracy difference was +1.81 percentage
points (95% video-group bootstrap interval +0.73 to +2.92; 320 groups). This is
not independent task-quality evidence: it is from the training-source OOF
diagnostic, with the same constructed labels and source family.

In a separate saved intervention, replacing an example's audio with audio from
another video carrying the same source label produced 55.25% Accuracy versus
54.98% original (2,581 matched rows / 320 groups). Original-minus-replaced was
-0.27 points (95% video-group interval -1.79 to +1.27), consistent with no
measurable effect on this diagnostic. Same-label replacement is not equivalent
to a same-physical-source intervention, so this alone cannot prove the model
ignores all audio information.

The balanced-set comparison also changed performance substantially on the
single-parent-visible stratum: 188/799 versus 34/799 correct, paired change
-19.27 points (95% group interval -22.53 to -16.21). The saved comparison is a
train-source diagnostic and is not evidence for a selected product model; it
does reinforce that the current task/readout behavior is sensitive to how
candidate labels are constructed.

## R2 consequence

Keep this readout and every external artifact as research history. Do not use
the exposed validation results to choose another head, loss, threshold, seed,
or modality weight. The current source-candidate task does not close R2-M.
Before another fusion fit, admit a fresh, rights-cleared, group-disjoint task
whose labels and controls demonstrate that both inputs are useful, including
an option/metadata-only baseline and matched unimodal, shuffled, stale, and
cross-group input controls. Existing independent PRs and data remain untouched.
