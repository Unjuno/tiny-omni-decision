# Fixed-model visual state and mask diagnosis

## Result and decision

The primary and separate fresh process **actually completed** with byte-exact
all24control prediction payloads, all training/validation latent arrays and
constants, metrics/statistics/intervals. No learning, selection or calibration update.

A single training-mean visual final state for every validation utterance preserves
**all5334slot answers** in each joint model: broadcast99.53%, ordered99.51%.
A fixed first training video or latent also preserves every answer. Matching visual
content is unnecessary for these observed predictions. This supports a nearly
constant learned-code explanation within this fixed-model development diagnosis,
not a universal statement about multimodal models or a new audio-only product.

In joint_ordered,100%of validation latent entries have abs(h)>=.99, average
across-example channel std1.401e-5. Its visual contribution to the first readout
affine layer has std2.639e-5 versus audio2.94249. Native visual inputs vary
(std1.08201); the strongly attenuated variation is in the trained joint readout.
The same frozen encoder's video-only model has latent std .61333 and affine
contribution std1.11055; cached-token reversal reduces48.01%→21.18%.
Do not diagnose an encoder failure from the joint branch collapse alone.

Zeroing the latent lowers joint ordered99.51%→97.43%; clearing only its presence
bit then gives97.34%. Thus nonzero learned visual state contributes to this
parameterization, but matched visual semantics do not account for the answers.
Equal registered parameter counts did not ensure equal active capacity in the
earlier audio comparison. No capacity/epoch/auxiliary-loss sweep follows.

Keep these scores and checkpoints in their scope. **No joint-Omni promotion**;
next require a task where audio and candidate videos supply complementary
evidence, with difficult source-matched distractors and unimodal controls.

## All fixed controls (selected checkpoints, unchanged calibration)

| Model | Control | Accuracy | NLL | Brier | ECE | Answer changes | Max raw logit change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint_broadcast | matched | 99.53% | 0.016100250 | 0.007526972 | 0.003492005 | 0 | 0 |
| joint_broadcast | input_zero | 99.42% | 0.019745944 | 0.009067220 | 0.005628808 | 17 | 3.64137 |
| joint_broadcast | input_train_first | 99.53% | 0.016100765 | 0.007534107 | 0.003568591 | 0 | 0.386696 |
| joint_broadcast | reverse | 99.53% | 0.016100250 | 0.007526972 | 0.003492005 | 0 | 0 |
| joint_broadcast | latent_train_mean | 99.53% | 0.016100770 | 0.007534089 | 0.003568685 | 0 | 0.385733 |
| joint_broadcast | latent_train_first | 99.53% | 0.016100765 | 0.007534107 | 0.003568591 | 0 | 0.386696 |
| joint_broadcast | latent_zero_present | 97.45% | 0.089336108 | 0.041880561 | 0.024967336 | 133 | 9.07314 |
| joint_broadcast | latent_zero_absent | 97.34% | 0.093655803 | 0.043869752 | 0.026059100 | 137 | 9.13443 |
| joint_ordered | matched | 99.51% | 0.016035771 | 0.007502696 | 0.003078396 | 0 | 0 |
| joint_ordered | input_zero | 97.79% | 0.075870738 | 0.035567185 | 0.021663874 | 119 | 8.69106 |
| joint_ordered | input_train_first | 99.51% | 0.016035702 | 0.007502661 | 0.003078374 | 0 | 0.000251532 |
| joint_ordered | reverse | 99.18% | 0.026851556 | 0.012106130 | 0.008584146 | 34 | 5.87193 |
| joint_ordered | latent_train_mean | 99.51% | 0.016035726 | 0.007502673 | 0.003078391 | 0 | 0.000205278 |
| joint_ordered | latent_train_first | 99.51% | 0.016035702 | 0.007502661 | 0.003078374 | 0 | 0.000251532 |
| joint_ordered | latent_zero_present | 97.43% | 0.087877704 | 0.041165793 | 0.025351513 | 135 | 8.96982 |
| joint_ordered | latent_zero_absent | 97.34% | 0.092399205 | 0.043230130 | 0.026932901 | 140 | 9.04009 |
| video_ordered | matched | 48.01% | 1.377505876 | 0.609485080 | 0.043730498 | 0 | 0 |
| video_ordered | input_zero | 19.10% | 1.879431156 | 0.822866605 | 0.049412105 | 4249 | 13.0655 |
| video_ordered | input_train_first | 19.25% | 2.476121719 | 0.997604650 | 0.314404131 | 4213 | 17.0305 |
| video_ordered | reverse | 21.18% | 1.966858244 | 0.849471623 | 0.129975621 | 4083 | 15.4964 |
| video_ordered | latent_train_mean | 18.80% | 1.919298303 | 0.834320693 | 0.073882453 | 4418 | 11.3621 |
| video_ordered | latent_train_first | 19.25% | 2.476121718 | 0.997604650 | 0.314404131 | 4213 | 17.0305 |
| video_ordered | latent_zero_present | 18.37% | 1.845722912 | 0.810476516 | 0.017235012 | 4427 | 13.1665 |
| video_ordered | latent_zero_absent | 18.65% | 1.845351003 | 0.810364401 | 0.013122398 | 4385 | 13.2226 |

`input_zero` is the standardized pooled training mean repeated16times, not
a natural blank video. `input_train_first` uses frozen training index0.
Latent controls intervene at the existing readout boundary. Zero-present versus
zero-absent holds audio/zero latent fixed and changes only the visual-presence bit.
Mask/latent interventions are OOD; no natural missingness robustness claim.
Reversal reorders16cached tokens, not raw-frame re-encoding; it diagnoses
readout order dependence, not full physical temporal reasoning.

## Integrity, uncertainty and preservation

Models fixed in advance: joint_broadcast23/joint_ordered23/video_ordered18.
All4080training and889development utterances, six full vocabularies/51options
retained. Train means use FP64 accumulation→FP32, no targets. First training
source:`s11_l_bbab2n`; this identity is recorded only,
never a neural feature. Existing398calibration temperatures are unchanged.
Native/decomposed readout logits are exact before any intervention.

Paired whole-speaker10000bootstrap(seed17, linear percentile95%) of fixed
train-mean-latent minus matched is in the adjacent JSON, along with all per-slot
metrics. Accuracy/whole-command differences are exactly0on these samples.
Nine observed development speakers, selected weights, single seed and artificial
controls limit interpretation. No population-equivalence claim. NLL changes near
1e-8–1e-6 are not presented as practically useful quality gains.

106355registered parameters per sequentially loaded head, unchanged native
encoder122573568/normalization2304 composition; encoders never loaded here.
CPU i7-12700H,4threads/interOp1, FP32/deterministic, Python3.11.9,
Torch2.6.0+cu124/NumPy2.4.6. Requires_grad flags retained for parity;
eval/no_grad, no optimizer, gradients or updates. CPU tests overlapped the
primary inference, so these are observed shared-PC wall times.

| Stage | Seconds | Peak CPU GiB | New output MB | Receipt SHA |
| --- | --- | --- | --- | --- |
| lombard-visual-state-diagnosis-01 | 22.963 | 2.080 | 11.842 | `25076e30d398defcfcb5c7a5123c27aed371703354c3902bdd927e898d56897f` |
| lombard-visual-state-replay-01 | 22.767 | 2.081 | 11.842 | `c1ac0b1d699d623048388dfe61f2166aead7608f9449a12f198574b5483632a0` |

Source commit:`c6970fe65605a367de9fabd50e4d436cc40939da`. Exact commands, runtime, checkpoint/
corpus/cache hashes, state statistics and controls in
[small results JSON](R2_LOMBARD_VISUAL_STATE_RESULTS_20261015.json).
Both new output directories remain under`C:/CodexArtifacts/pretrained-reuse-r2/`.
All source/runtime/checkpoint/receipt/control inputs are hash-bound and rechecked
before/after execution, replay and publication. Resource guards600s/3GiB/64MiB/
1GiB reserve held. No cloud/GPU/old8GiBguard changes, Teacher/sealed changes
or legacy artifact overwrite. Weights/media/latent arrays stay outside Git.

The user-withdrawn standalone <=10M audio program remains withdrawn. This
diagnosis informs the required native-audio component of the main <=200M Omni
research; it does not substitute another product goal. Four-modality95–98%,
genuine joint/OOD/condition reuse and deployment gates remain unmet.
