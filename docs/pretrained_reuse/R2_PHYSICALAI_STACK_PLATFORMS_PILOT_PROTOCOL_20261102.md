# PhysicalAI stack-platforms video + state action-choice pilot protocol

## Frozen task and scope

This protocol is fixed before feature extraction, probe fitting, or inspection
of validation labels/predictions for the selected 700/250 pilot. It tests a
narrow text-free decision proxy, not task success or a product capability:

> Given the last eight synchronized world-camera frames and eight aligned
> 81-dimensional object/robot state rows, assign probabilities to eight
> fixed action prototypes for the demonstrator's recorded action three frames
> (100 ms at 30 fps) after the observation window.

The eight candidate actions are deterministic clusters of future 7-D actions
fit using train episodes only. Each label is the closest prototype to the
recorded future action after per-dimension train-only standardization. This
turns the dataset's continuous action into a fixed-option decision; it does
not mean the selected action is optimal or successful.

## Frozen data and model protocol

- Source revision and CC BY 4.0 status are recorded in the
  [source screen](R2_PHYSICALAI_STACK_PLATFORMS_SOURCE_SCREEN_20261102.md).
- Use split manifest v3 and only the fixed pilot: 700 train episodes and 250
  validation episodes; four windows per episode (2,800 train and 1,000
  validation queries). Group unit is publisher episode ID. Reserved episodes
  are never loaded.
- Window endpoints are the deterministic 20%, 40%, 60%, and 80% positions in
  each episode's valid range. Input state is rows `t-7..t`; target is action
  row `t+3`. Verify Parquet timestamp/frame-index agreement and decoded frame
  rate before extracting features.
- Video input: frozen DINOv2-base at revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, FP32. For the primary video
  feature, average per-frame CLS embeddings across each eight-frame window;
  retain the last-frame feature as a predeclared control. This mean pooling
  discards explicit frame order and is not a temporal-reasoning test.
- Sensor input: flatten the eight 81-D state rows to 648 features.
- Fit train-only action normalization, the deterministic eight-prototype
  codebook, per-view feature normalization, and linear eight-class softmax
  heads. Fixed optimizer is full-batch LBFGS, `max_iter=200`; seed 17. No
  calibration fitting, validation checkpoint selection, hyperparameter search,
  or extra seeds.
- Compare on identical validation queries: state only, eight-frame video
  mean, last-frame video, and joint video+state. Retain train-class-prior,
  episode-misaligned video, missing-video, and missing-state controls. Missing
  inputs are replaced with the respective train feature mean under the already
  fitted joint head; no control is refit.
- Report Accuracy, NLL, multiclass Brier, ECE-10, class counts, per-class
  Accuracy, and train metrics for each arm. Use a paired 2,000-resample
  episode-cluster bootstrap (seed 17) for joint-minus-state, joint-minus-video,
  and the missing/shuffled controls. Four windows from one episode remain one
  bootstrap cluster.

## Decision rule and limits

Both-input contribution is supported only if the joint model exceeds the best
single-input Accuracy and the paired episode-cluster 95% interval for that
difference excludes zero; missing and episode-misaligned controls must also
show that predictions use the corresponding input. Mixed controls, inadequate
class support, or intervals spanning zero are `UNCERTAIN`, not a pass. No
validation result from this pilot may trigger task, candidate-count, feature,
or optimizer changes on the same IDs.

The complete normalized-frame screen checks exact SHA-256 equality after RGB
conversion and bilinear resize to 32x32. It is not a near-duplicate or
scene-identity test. The upstream metadata exposes no scene/layout/environment
ID, so every eventual result is only episode-held, same-source development
evidence. Keep the source `REVIEW_NOT_ADMITTED`; do not access reserved media
or promote a candidate based on this pilot.
