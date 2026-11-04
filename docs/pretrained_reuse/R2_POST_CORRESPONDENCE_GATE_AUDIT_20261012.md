# R2 gate audit after the fixed correspondence experiment

This is a requirements audit, not a new model evaluation. Source state5da858a.
Preserve the <=200M **pre-quantization** resident-parameter constraint and the
95–98% per-modality/fusion aspiration. No aggregate score replaces either gate.

| Required capability | Existing evidence | Remaining gap / next admissible evidence |
|---|---|---|
| Conditional text decisions | [Coverage training](R2_TEXT_COVERAGE_RESULTS_20261012.md) reaches63.58% original /84.08% curriculum at selected1792; [transfer](R2_COVERAGE_TRANSFER_RESULTS_20261012.md) retains domain-word/range failures | Generalization across conditions and formats, not additional epochs on the same observed validation |
| Image decisions | Synthetic CLEVR-4 task probes and native runtime replays | Representative natural images, counting/relations and condition changes; task-specific successes are insufficient |
| Temporal video decisions | CLEVRER controls and later token-readout comparisons; existing runtime accepts raw video | Useful condition-sensitive temporal evidence and representative held-out tasks; framewise image success is insufficient |
| Native speech and event audio | Whisper transcript-choice path and separate DyMN probes | Speech comprehension, event decisions and conditions in one validated product; transcript matching is a narrow task |
| Joint audio/video | [Fixed correspondence comparison](R2_AV_CORRESPONDENCE_RESULTS_20261012.md): selected51%, untrained25%, strong train memorization | Correspondence does not establish semantic joint reasoning; do not call it a multimodal product benchmark |
| Sensor and simultaneous input handling | [Resident replay](R2_RESIDENT_NATIVE_PATHS_20261012.md) preserves sensor/image/video/speech outputs | Episodes in the replay are unrelated, not synchronized fusion evidence; static central-label future-evidence scope remains |
| Screen captures | Prior screen experiments remain separate | No accepted general screen decision pathway; do not count image taxonomy as screen support |
| One compact resident model |169,973,492-parameter functional slice | Excludes standalone text/event-audio/screen/fusion. Neither adding unrelated scores nor an inventory sum proves a unified model |
| Custom JSON probabilities | Existing deterministic serializers and native replay | Serialization works independently of whether the probabilities are accurate/calibrated on required tasks |
| Compression/release | Storage estimates and historical experiments | Current quality does not authorize R7 compression or R8 final audit; retain sealed data untouched |

## Completed experiments that must not be repeated as new work

- Text final-block budget scaling, all-block coverage, numeric/lexical coverage
  and fixed coverage-transfer controls already exist. Read their measured results
  before suggesting the same intervention. The newer coverage candidate is not
  a drop-in replacement for shared image/video heads trained on another encoder.
- The100-epoch AV correspondence run is complete, including controls, reloads
  and clustered uncertainty. Do not restart it, change its selector to Accuracy,
  or tune temperatures/capacity on the same200 recordings.
- Concurrent residency and JSON parity already have real measurements. Repeating
  those replays alone cannot close the quality gate.

## Next research decision

The missing evidence is **condition-sensitive semantic decisions**, including
useful synchronized modalities. Before another fit, define a prospective task
contract specifying observations, optional criterion, candidate semantics,
independent source grouping, missing-modality behavior and paired condition
changes. Preserve all examples and report unimodal / joint / evidence-shuffle
controls. A model must change its answer when the correct decision changes,
and preserve it for irrelevant input changes; simple output change is not success.

Use rights-approved sources and existing fixed encoders first. Reuse saved media
only as explicitly observed development material; it cannot become a new blind
audit. Establish whether the required label/evidence relation is actually present
before training. Neither a new dataset name nor a new prompt grants independence.

Do not infer that all model families have reached a ceiling. Current evidence
does justify closing these particular repeated fitting paths and leaving shared
Lens/merging/quantization gated. Goal remains active: no all-modality95% result,
release candidate or completion claim exists.
