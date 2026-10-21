# Native four-video speech correspondence: frozen protocol

Bounded extension of existing AV correspondence evaluation, following measured
visual-state diagnosis at3b8578c. Prior goal turn is progress: actual fixed-model
diagnosis/replay, published tests/CI and parent-pool metadata audit. User permits
autonomous public ALLOW/local work. No new model fitting, encoder changes or
standalone audio product. This controlled task is not generic Omni competence.

Question: which of four videos depicts the supplied native spoken command?
Audio/query plus candidate videos, optional criterion, four opaque custom-ID
choices/JSON probabilities. Reuse publisher-approved CC BY4.0 native Lombard
source, exact4080train/398calibration/889observed development parents, grouping
and all source/media/model/cache hashes. No sealed audit, new frames, pruning
of difficult parents or reduction of original51semantic options/25letters.

## Catalog and source control, fixed before scores

Two catalogs: random versus hard(near-command). Every original parent anchors
one quartet of four distinct actual six-slot commands. All members come from
the same split, speaker and p/l recording condition: speaker/background cannot
identify the matching command. Exclude same-command negatives; choose a stable
seed17SHA representative for each eligible command, then three negatives by
seed17SHA order(random) or smallest six-slot Hamming distance followed by the
same hash(hard). No prediction/performance-based negative selection. Randomize
the four candidate positions by independent seed17SHA. Fail if fewer than three
distinct commands; retain every parent rather than skipping such cases.

For each fixed quartet, evaluate all four members as audio query with candidate
videos/order unchanged. Correct positions0/1/2/3occur exactly once. This prevents
a video-only centrality/selection shortcut in hard quartets. Catalog counts:
16320train/1592calibration/3556development queries per policy, no training updates.
Track unique parents/audio/video, unique unordered quartets/query contexts,
repeated contexts/presentations, realized speaker/condition/distance counts.
3556questions are correlated observations from889assets/nine development
speakers, not3556new independent held-out sources. Same physical clips may be
reused within a split; none cross splits. Target/label/source IDs never enter
the neural readout or matching function, only catalog eligibility and audits.

## Frozen model and rule

Use already selected temporal audio24 and video_ordered18 weights/normalizers
and original calibration-only six-slot temperatures. Preserve those artifacts.
Verify full train/calibration/development model reload against cached raw logits
before matching. Their learned51class posterior vectors are neural features;
no ground-truth transcripts/labels as matching input. All native frames retained.

For audio a and video v, score is the SUM over six slots of
log(sum over full native options k of P_audio(k|a)*P_video(k|v)). This fixed
product-of-slot-agreement is an independence approximation, not claimed joint
probability. Stabilize log-sum-exp without dropping options.
Keep NLL in the log domain without clipping; stored probabilities may underflow
to zero while their finite log probabilities still record the full loss.
No new projection, hand-selected modality/slot weights, label injection or
validation-based search.
One new scalar temperature for four-choice softmax: earliest minimum calibration
NLL on combined random+hard3184queries, fixed17geometric temperatures0.25..4.
Report raw T1 and calibrated metrics; no development-driven temperature choice.

Every policy reports raw/calibrated train/development Accuracy/NLL/Brier/ECE,
mean confidence, class counts and source accounting. No quality early exit.
Missing either required input yields explicit uniform four-choice probabilities,
a prototype no-evidence policy, not learned missingness robustness or proof about
all possible unimodal models. Video-only task inputs are identical across the
four balanced roles; no static deterministic video-only answer can exceed25%.

## Fixed interventions and reproducibility

Development controls per policy: matched, audio absent, videos absent, audio
query cyclic swap, video content cyclic swap under unchanged IDs. Retain original
targets for change diagnosis and separately report corrected targets after swaps;
corrected aggregate metrics should equal matched by balanced role permutation.
Distinguish video-content/ID swap from legitimate option reorder: reorder videos,
custom IDs and targets together, realign JSON by ID, require exact probabilities
and selected IDs on all7112policy-query observations. Save IDs, indices, original/
corrected targets, raw scores/probabilities/predictions and calibrated metadata.

Paired whole-speaker bootstrap10000replicates/seed17/linear percentile95% of
matched-minus-uniform for each policy, preserving every quartet/role in its
speaker cluster. No row bootstrap. Random/hard use different candidate/query
distributions; do not interpret their difference as a paired model improvement.
Nine observed development speakers/old checkpoint selection/single seed limit
inference. No blind quality, per-modality95–98%, merging or product promotion.

CPU4threads/interOp1/FP32neural/FP64scoring/deterministic, encoders never loaded,
zero optimizer/gradient updates. Guard600s/3GiB peak/64MiB new output/1GiB disk
reserve; old GPU8GiBguard/unfinished run untouched. New output only. Bind all
source/runtime/checkpoint/corpus/receipt/catalog/code inputs before and after.
Freeze code before primary run; a separate process recomputes models, catalogs,
calibration, controls, metrics and intervals, requiring exact array/report parity.
All attempts/outcomes retained; publish small results/tests/CI without weights.

Registered composition: native encoders122573568 + two full106355heads + two
2304normalizers + twelve parent temperatures + one query temperature =122790899
parameters/coefficients. No pruning or shared normalization cost credited.
Only this AV task slice; complete four-modality package/latency/quality unmet.
