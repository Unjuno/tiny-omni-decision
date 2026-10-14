# Fixed RoboMNIST source-length shortcut diagnostic

Purpose: measure whether directory-only input file lengths can predict activity
labels before spending resources on media feature extraction. This is an offline
shortcut diagnostic, not a pretrained-feature Omni baseline, and not a native
audio/video/CSI quality result. The full goal remains unchanged.

## Fixed cohort, split and fields

Use the complete pinned replay02 inventory (SHA256
`b34bc54f57d3c434947d673ed32cd9a9c6e4f82bfc10f9b02213b397a76a29a4`),
all60 classes/all2,135 physical repetitions. Revalidate all70 parent pins and
every directory's lexical/group binding. No new payload, GT or audit-set access.

The same numeric repetition block, across all classes/views, must share a split.
Keep already-observed Rp01 on the train/development side. Among other repetition
numbers, order by SHA256 of `robomnist-rp-block-v1:17:{number}`; validation takes
the first ceil(number_of_all_blocks/4). All others are train. No row is omitted,
no seed/partition retry. These are development split proposals only: true
collection-session independence, content uniqueness and eventual sealed/OOD
admission remain unverified. Equal Rp numbers are conservatively grouped;
unequal Rp numbers are not proven separate sessions.

Allowed diagnostic predictors: log1p of uncompressed file sizes of camera1/2/3,
microphone1/2/3 and CSI, in this fixed order. Names, CRCs, archive sizes, class
indices, uncertainty, repetitions and positionGT size are forbidden predictors.
Names identify source bindings and labels only. All seven inputs remain in one
group. Reject missing/duplicated/substituted streams or duplicate class/Rp rows.

## Fixed estimator and outputs

Four centroid arms: camera lengths, microphone lengths, CSI length, all seven.
Compute feature mean/std from train only (std below1e-12 replaced by1), then
the mean vector for each of all60 classes. Fixed logits are minus one half of
squared Euclidean standardized distance to each centroid; stable log-softmax,
temperature1, equal class priors. No regularization/schedule/search/calibration.
A fifth train-count frequency prior is the no-input control.

Save every train/validation row and all60-class log probabilities. Aggregate
them without refitting to four fixed supplied-option tasks: full60-class
activity; robot2; velocity3 (high,med,low); trajectory10 (digits0–9).
NLL is computed from stable log probabilities without probability clipping;
Brier and15-bin top-label ECE use exponentiated probabilities. Keep train and
validation metrics separate; never treat four questions per row as independent
recordings. No target-dependent option count reduction or difficult-row removal.

Report paired Accuracy difference from the prior using2,000 percentile bootstrap
draws, seed17, resampling complete validation repetition blocks; intervals are
development diagnostic uncertainty, not blind/generalization evidence. Report
all arms, not only the best. A high score is shortcut potential, not proof that
the source is unusable or that legitimate physical evidence is absent.

CPU only, no learned encoder/head/checkpoint, no downloads/dependency upgrades.
Execution bound120s,2GiB peak CPU,16MiB output,1GiB disk reserve. New output
directory; preserve all preceding artifacts and the separate8GiB GPU guard.
Commit protocol/tests/estimator before actual fitting; save effective config,
source/consumer hashes, source commit, full counts and terminal reason. CPU/Ruff/
manifest checks and independent publication review before normal push/CI.
