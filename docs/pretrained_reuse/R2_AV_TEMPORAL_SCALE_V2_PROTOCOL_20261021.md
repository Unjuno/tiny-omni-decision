# Fresh AV temporal matching scale experiment — frozen protocol

## Question

Does increasing unique training video groups from 96 to 224 reduce the rapid
train/validation divergence seen in the earlier 96-group temporal AV diagnostic,
and does preserving eight VideoPrism frame vectors help relative to mean
pooling under either data budget?

This is a narrow task diagnostic, not general audiovisual understanding or
product evidence. The target is to choose the synchronized audio window from
two same-video, same-label event windows for an eight-frame video interval.

## Frozen data contract

- Source: Perception Test pinned annotation revision
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`, official TRAIN only.
- 224 unique videos form the large training cohort; a deterministic nested
  subset of 96 videos forms the small cohort; the same additional 64 videos
  form validation for every candidate.
- All questions from a video stay in one split. Training labels are used only
  to build same-label negatives and stratify the two training budgets.
- The 224 train, 96 nested train, and 64 validation groups have no group
  overlap. The source selection excludes hash-verified prior exposure receipts,
  the prior temporal-match preflight cohort, and the first v2 temporal cohort.
- Validation is development-only, will be used for the fixed checkpoint
  selector, and is not blind or independent test data. No official test or
  sealed audit data are opened.
- Exact frozen manifest SHA-256:
  `9147e598a4ac9e8412204b472ac6b32a019517ff682b8587379f23605672dbef`.
- Data sizes and label counts are saved in the attempt's `task_manifest.json`.

## Candidate matrix

Fit six arms, all with seed 17 and a maximum of 2,048 full-batch AdamW updates:

| Training cohort | Audio-only | Pooled AV | Eight-frame LSTM AV |
|---|---:|---:|---:|
| Nested 96 groups | yes | yes | yes |
| Full 224 groups | yes | yes | yes |

Model definitions are those in
`scripts/fit_perception_temporal_match_sequence_readouts.py`: audio-only option
scorer (12,353 parameters), and equal-size pooled / sequence AV scorers
(50,625 parameters each). The sequence scorer conditions an LSTM initial state
on each audio option and attends over the eight VideoPrism frame features.
The pooled control averages those same eight features to one timestep. The
VideoPrism (114,365,184 parameters) and EfficientAT (1,961,239 parameters)
encoders stay frozen.

Fixed optimizer settings: AdamW, learning rate 0.002, weight decay 0.001,
constant LR, full batch, 2,048 updates per arm. Evaluation occurs at step 0 and
every 32 updates. There is no early stopping: all six arms finish the fixed
budget, while the selected checkpoint is the earliest minimum validation
categorical NLL. Train-only normalization is fit separately for the 96- and
224-group training cohorts. No hyperparameter, architecture, or seed selection
is allowed after validation results are observed.

Record train and validation Accuracy/NLL/Brier/ECE-10, the complete training
curve, selected and final steps, per-row predictions/probabilities, counts,
and paired 5,000-resample video-group intervals (seed 17). On each selected
sequence checkpoint, additionally evaluate missing-video, missing-audio, and
time-reversed inputs. These are sensitivity controls, not evidence of a trained
counterfactual capability.

## Selection and interpretation

The principal scale comparison is 224 versus 96 groups for the same model arm
on the same 64 validation groups. The temporal-architecture comparison is
sequence versus pooled AV at each fixed training size. Report Accuracy and
NLL together; do not treat validation-selected Accuracy as blind evidence.
Train rows are repeatedly reused for full-batch updates, so optimizer updates
must not be described as new unique examples. State whether the curves show
continued learning, memorization, or a mixed result; do not infer a model-family
ceiling from this one source/task/seed.

If an input asset fails its pinned size/ETag/ZIP CRC, media identity gate, or
feature finite/hash check, stop before fitting and record the exact failure.
No fallback data or extra seed is authorized by this protocol.
