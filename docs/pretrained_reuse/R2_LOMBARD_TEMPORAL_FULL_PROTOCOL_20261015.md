# Full execution admission and evaluation contract

Extends the immutable temporal protocol committed at20f1bdc. Its smoke-only CLI
and all executed code/config/test/spec files stay unchanged. Separate full
controller reuses its exact fit functions and requires the verified five-arm
four-training-row smoke, identical runtime/source pins and fixed effective config.

Training starts fresh for all five arms, identical seed17 initialization and
24orders of all4080 training examples,768updates per arm. No selection from
controls, intervals, calibration Accuracy or final epoch; earliest minimum raw
development macro NLL only. All24epochs run unless safety/integrity/resource fails.
Save raw/calibrated selected and raw final predictions and separate best/final
weights. No strict-resume claim. Actual training/validation consumption recorded.

Before full fitting, bind separate evaluator/helpers/tests and this contract.
Full trainer success means five fits completed, not full comparison verified.
Then a separate CPU process must reload all best/final weights and train-only
normalizers, regenerate all train/validation and selected calibration predictions
byte-exactly, verify all saved epoch predictions/metrics, selector, temperatures,
sampling/LR/gradient logs and checkpoint/optimizer counts. Keep requires_grad
flags as during training for numerical parity but use eval/no_grad, no optimizer
or weight update; assert unchanged model tensors. Encoders are never loaded.

Selected joint_broadcast/joint_ordered controls, all889fixed development rows:
matched, reverse16cached visual tokens (not raw-frame re-encoding), samecondition
different-speaker bijective video donor, audio donor, absent audio and absent
video. Absence is unseen-mask OOD, not evidence of supported missingness. Donor
labels never used as targets. Save original row identities/targets/probabilities,
donor mapping and control metrics. Preserve all original51options; for every
selected-arm observation, verify JSON probabilities after reversing option order.

Speaker-group paired bootstrap10000replicates/seed17/linear percentile95%:
video_ordered minus video_broadcast, joint_ordered minus joint_broadcast, each
joint minus audio, with selected calibrated scores. Nine observed-development
speaker groups, model selection and single seed limit inference. Compare raw
scores/final endpoint separately and report24point learning curves/train gap.
No blind audit/general Omni95–98% claims or product promotion.

Same CPU4threads/FP32/deterministic bounds for full fit7200s/3GiB/128MiB/1GiB
disk reserve. Evaluation uses its own new directory and same safety bounds;
do not change old GPU guard or restart interrupted legacy runs. Read-only source
and training artifact hashes are verified before and after evaluation. All
negative outcomes and any incomplete attempt are retained; no hyperparameter retry.
