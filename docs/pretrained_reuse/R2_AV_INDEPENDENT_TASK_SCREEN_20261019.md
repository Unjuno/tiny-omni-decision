# Independent audio-video task gate — 2026-10-19

## Decision

No newly screened source currently meets all three gates for another AV
quality-selected run: (1) usable media rights for this product-research path,
(2) independently grouped train/validation evidence, and (3) target labels that
establish complementary audio/video information or support valid
counterfactual controls. Do not launch another GPU fit against the existing
validation sets. This is a data/task admission decision, not a claim that the
model or architecture has reached a ceiling.

The user's HotUGPU scheduling path remains available for the next admitted
experiment. The current RTX 3080 was idle at the read-only check; no new GPU
job, data acquisition, model update, audit access, or process intervention was
performed in this screen.

## Candidate disposition

| Candidate | Task fit | Rights / split finding | Decision |
|---|---|---|---|
| AVQA | Manually authored multiple-choice questions include `View`, `Sound`, and `Both` relations. | Official terms restrict use to personal/classroom contexts; videos originate from VGG-Sound/YouTube. The source does not establish that every `Both` question truly needs both inputs. | REVIEW; do not ingest. See [source screen](R2_AVQA_SOURCE_SCREEN_20261019.md). |
| AVA ActiveSpeaker | Face-track-level speaking/audibility could test voice-to-visible-person association. | Original YouTube media rights remain unresolved. The linked validation archive contains test-named members and one row was inadvertently parsed; quarantine it. Train/validation integrity and label-schema reconciliation are unresolved. | REVIEW; do not use the contaminated archive. See [source screen](R2_AVA_ACTIVE_SPEAKER_SOURCE_SCREEN_20261018.md) and [label audit](R2_AVA_ACTIVE_SPEAKER_LABEL_AUDIT_20261019.md). |
| MAVD-traffic | Synchronized traffic A/V with sound-event intervals. | CC BY 4.0 metadata, but train/validation overlap on conservative site-day groups; labels do not identify the visible source of each sound. Privacy review is incomplete. | REVIEW; no both-required task admitted. See [source screen](R2_MAVD_TRAFFIC_SOURCE_SCREEN_20261019.md). |
| UnAV-100 | Time-localized audio-visual event labels could support event-presence decisions. | The official page says CC BY 4.0 while reserving source-video copyright; a mirror advertises CC BY-NC-SA 4.0. Media, exact split groups, and usable rights remain unresolved. | REVIEW; do not ingest. See [source screen](R2_UNAV100_AV_EVENT_SCREEN_20261011.md). |
| AVE / AudioSet | Event labels and train/eval partitions are available; AVE contains audio and video. | AVE audio/video assets are YouTube-sourced. AudioSet's labels describe sound classes, not visual truth or joint necessity; its evaluation split is already distinct and is not a paired product task. | Not admitted as a joint Decision task. See [DCASE AVE catalog](https://dcase-repo.github.io/dcase_datalist/datasets/sounds/ave.html) and [official AudioSet page](https://research.google.com/audioset/download.html). |
| AV-SyncBench | Pairwise temporal/semantic synchronization probe over 3,269 videos and 38,390 generated evaluation samples; no model training is part of its protocol. | The official repository says dataset media may retain original-creator rights and does not declare a separate license for its adapter. It is a synchronization benchmark, not a broad conditional Decision task. | Potential inference-only research reference after media rights/terms are resolved; not admitted to training or product-quality evaluation. See [official repository](https://github.com/Duck-Boss/AV-SyncBench). |
| STARSS23 | Licensed synchronized environmental A/V and room-grouped development recordings. | Existing frozen comparison had only two validation rooms; Audio-only outperformed the joint readout on Accuracy, NLL, Brier, and ECE. Video removal improved the joint model. The official dev-test remains unopened. | Preserve the negative result; no further tuning on the same rooms. See [comparison](R2_STARSS23_COMPARISON_20261011.md) and [split record](R2_STARSS23_SPLITS_20261011.md). |
| Greatest Hits | Same-recording audio/video matching has balanced unimodal target marginals and valid synthetic pair swaps. | The author page states CC BY 4.0, but the released label schema and actor/site identity are not independently certified. Its fixed same-source task has only 28 validation pair clusters and severe train/validation overfit; the scoped source review does not certify a both-required task. | Retain as a low-power exploratory development baseline; no capacity expansion or repeated validation search. See [source review](R2_GREATEST_HITS_SOURCE_REVIEW_RESULTS_20261017.md), [protocol](R2_GREATEST_HITS_AV_MATCH_PROTOCOL_20261018.md), and [results](R2_GREATEST_HITS_AV_MATCH_READOUT_RESULTS_20261018.md). |

## Existing fixed-checkpoint diagnostic is complete

The authorized E0G1 replacement diagnostic already completed as attempts 11,
14, and 15. All three successful runs reproduced the same 44,375 prediction
rows, 17,750 exact self-replacements, and sensitivity digest. They made zero
optimizer or encoder updates and did not load calibration or sealed-audit
data. The result establishes output sensitivity to the replacement operation,
not correctness or causal use of valid paired evidence. Attempt 09 was an old
incomplete directory and was correctly preserved rather than overwritten; the
authorized later rerun used fresh attempt directories. Do not repeat this
diagnostic absent a changed, truth-bearing task.

## Next admission contract

Before allocating GPU time to a new AV fit, require one source packet with:

1. a pinned official release and per-component license/rights scope;
2. row-level train/validation/test identities grouped by the underlying video,
   recording session, and available speaker/site identity;
3. enough independent train and validation groups for the declared uncertainty
   target, with exact media/content overlap checks;
4. labels whose truth depends on both streams, or predeclared counterfactual
   labels that remain valid under each intervention;
5. identical frozen input/target manifests for the unimodal and joint controls.

If the source does not meet those checks, keep it REVIEW and do not spend GPU
time trying to manufacture a positive fusion result. When a source does pass,
use HotUGPU's cooperative reservation wrapper; never stop or signal Orbitune.

This gate leaves the 100–200M pre-quantization parameter target, 95–98% task
goal, four-modality requirement, and all R2 exit criteria unchanged. No
product Teacher is promoted and no sealed evaluation is opened.

## References

- [Official AVQA project and terms](https://mn.cs.tsinghua.edu.cn/avqa/)
- [Official Google AVA project](https://research.google.com/ava/)
- [Official AVA download and label links](https://research.google.com/ava/download/)
- [Official AudioSet license and split description](https://research.google.com/audioset/download.html)
- [DCASE AVE catalog](https://dcase-repo.github.io/dcase_datalist/datasets/sounds/ave.html)
- [AV-SyncBench official repository and rights notes](https://github.com/Duck-Boss/AV-SyncBench)
- [Official UnAV-100 project](https://unav100.github.io/)
- [Zenodo MAVD-traffic record](https://zenodo.org/records/4741232)
- [Zenodo STARSS23 record](https://zenodo.org/records/7880637)
