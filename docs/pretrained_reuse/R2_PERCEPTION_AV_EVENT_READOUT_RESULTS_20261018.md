# Perception AV event-source readout — 2026-10-18

## Scope and outcome

This fixed development experiment asks which of several annotated visual
objects is the source of a sound event in the same video. The frozen encoders
are DINOv2 and CLAP. It is a narrow audio-to-object correspondence task, not a
general Omni decision benchmark. No sealed/test data was opened, and the result
does not qualify the four-modality 95–98% objective or product deployment.

The first real fit exposed two pipeline defects before producing a result:
the feature extractor copied the acquisition-plan `train` split directly,
leaving all 300 training videos outside `fit`/`selection`; and the metric
aggregator assumed all videos had the same number of candidate objects. Both
were corrected. The original feature arrays and raw manifest were retained.
The corrected manifest changes only each row's `split` value, verified by
comparing every other field and every referenced array hash.

## Frozen data and model inputs

The acquisition plan SHA-256 is
`9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25`.
Deterministic SHA-based grouping assigns 300 official TRAIN videos to 240 fit
and 60 selection videos. The other pinned groups are 100 calibration TRAIN
videos and 400 official VALID development-validation videos. The 1,719 event
rows are distributed as follows:

| Split | Videos | Events |
|---|---:|---:|
| Fit | 240 | 525 |
| Selection | 60 | 105 |
| Calibration | 100 | 188 |
| Development validation | 400 | 901 |

All video IDs and media SHA-256 identities are split-disjoint. No test archive
was acquired. Candidate count varies substantially: development-validation
events have 2–47 candidates; fit has 3–34. Thus a single fixed-choice chance
rate is inappropriate. DINOv2 and CLAP remain frozen (22,056,576 and
28,190,872 parameters, respectively). DINOv2 CLS features are used for each
candidate; this is the runner's first pooling choice, not a measured optimum.

Raw feature manifest:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-features-02\features.jsonl`
(SHA-256 `720e817cbd2f9a7eeb4124786bd027f8b5dbbebe713fe32031cb1b50d54f9cd1`).
Final split-correct manifest:
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-features-02\features-final.jsonl`
(SHA-256 `d92e4c91c236c69548f8fa0b58826e8afc853372ebf7a59b1e64567034f8df57`).
Finalization receipt SHA-256:
`fc9d62d7a59399e3d73dfd18599e6347bd654f3080b93b7b4f16531619401da6`.
The sorted selection-video ID hash is
`665b9fcd56bef808fa36709062074485dc875107e07007c0f268e2a69202742b`.

## Fixed fitting protocol

Source commit before this change: `d1dba798ac5bdfd101be6a1f0d59ddb7a850c6ef`.
Both conditions use seed 17, AdamW at 0.001, weight decay 0.01, hidden size
128, at most 100 epochs, patience 20, and absolute selection-NLL `min_delta`
0.0001. Fit events alone update weights; video-disjoint selection NLL selects
the earliest best checkpoint. Best and final weights are separate and best
reload is checked. Calibration and development validation are scored once on
the selected checkpoint; neither is used to choose it. Calibration metrics
below are held-out scores, not post-hoc calibrated probabilities.

The follow-up condition was predeclared by matching video-only readout size to
the joint readout. A hidden size of 339 gives 246,115 trainable parameters,
within 0.22% of joint's 246,658. It uses the same seed, data, selector and
optimizer schedule; only hidden size changes from the original video-only arm.

The conditions are a trainable joint audio+visual readout and a visual-only
readout. Their capacities differ: 246,658 and 65,921 trainable parameters.
Consequently, a joint-vs-video difference cannot be attributed to audio alone.
Uniform probabilities and a fit-derived candidate-position prior are reported
as controls. The position prior uses add-one counts within candidate-count
groups, estimated only from fit rows.

Attempt `perception-av-event-readout-joint-01` performed the first fit epoch
(525 optimizer updates), then stopped when selection evaluation encountered
variable-length candidate logits. It saved no checkpoint. Its failure receipt
is preserved. The metric code now scores each event's own option vector without
padding. The corrected joint, original video-only and capacity-matched
video-only runs completed on CPU; no GPU training was used. Run outputs are
outside Git.

## Results

Metrics are Accuracy / NLL / Brier / ECE. Best-selection metrics choose the
checkpoint; calibration and development-validation metrics are fixed held-out
comparisons.

| Condition | Best epoch (completed) | Trainable params | Selection | Calibration | Development validation |
|---|---:|---:|---|---|---|
| Joint audio+visual | 2 (22) | 246,658 | 0.324 / 1.983 / 0.788 / 0.136 | 0.431 / 1.832 / 0.753 / 0.067 | 0.388 / 1.897 / 0.763 / 0.058 |
| Visual only, hidden 128 | 8 (28) | 65,921 | 0.352 / 1.974 / 0.805 / 0.085 | 0.335 / 2.193 / 0.850 / 0.106 | 0.327 / 2.178 / 0.829 / 0.083 |
| Visual only, capacity-matched hidden 339 | 3 (23) | 246,115 | 0.362 / 1.995 / 0.787 / 0.093 | 0.410 / 1.918 / 0.783 / 0.110 | 0.363 / 1.990 / 0.797 / 0.117 |
| Uniform over each event's candidates | — | 0 | — | — | 0.091 / 2.669 / 0.924 / 0.017 |
| Fit candidate-position prior | — | 0 | — | — | 0.072 / 2.754 / 0.938 / 0.054 |

On the 901 development events, joint minus the smaller visual-only Accuracy is
+0.061 (95% video-cluster percentile interval +0.024 to +0.099); NLL difference
is −0.282 (interval −0.368 to −0.198). That comparison confounds modality and
capacity. A predeclared follow-up matched visual-only readout used hidden size
339: 246,115 trainable parameters versus 246,658 for joint (0.22% difference).
Joint minus matched visual-only Accuracy is +0.0255, but its video-cluster
interval (−0.0079 to +0.0588) includes zero. Joint NLL is lower by 0.0935
(interval −0.1551 to −0.0335); Brier and ECE are also lower for joint on these
same development IDs. Each paired bootstrap resamples 400 video IDs with
replacement for 10,000 draws using seed 17. The matched model size was fixed
from the parameter equation before fitting; its checkpoint is still selected
only on selection NLL. The development videos have already been observed and
used in an earlier candidate comparison, so neither bootstrap is an independent
blind estimate. Candidate-count strata, probabilities and per-example outputs
are saved in both analysis artifacts.

Overfitting was visible. Joint training CE fell from 1.976 at epoch 1 to 0.001
at epoch 22 while the selection NLL rose to 6.979; selected epoch 2 had training
CE 1.517. Visual-only training CE fell from 1.986 to 0.279 by epoch 28 while
selection NLL reached 5.978; selected epoch 8 had training CE 1.231. Patience
stopped these runs; neither final checkpoint replaced the best checkpoint.
The matched visual-only arm also overfit: training CE fell from 1.986 to 0.188
by epoch 23 while selection NLL reached 7.685; selected epoch 3 had training CE
1.577. Its best checkpoint was reload-verified and kept separate from final.

| Artifact | Result SHA-256 | Wall time |
|---|---|---:|
| `perception-av-event-readout-joint-02/result.json` | `1a88c7fb791bd8edbc3be2642f53ff56fed832c12fbcb2beb4e4428c9fe96434` | 46.82 s |
| `perception-av-event-readout-video-only-01/result.json` | `66ea019f5259af72c7d49dfea3f618dfb3e03008c9175900f6bbeb2825fb1f20` | 28.39 s |
| `perception-av-event-readout-analysis-01/analysis.json` | `59858ee6a3c333167dbc19882ed3576cb61cd2cd8cb7af1f307ac9866b6c1699` | under 4 s |
| `perception-av-event-readout-video-only-matched-01/result.json` | `a602a3de51e0031628dfe55f04f3b5b5987725c357d6dab4e04801255e9d8354` | 33.87 s |
| `perception-av-event-readout-analysis-02/analysis.json` | `5504551590706dabb81006fdb70df14f2443fe4d48260ff13018aa9e029a7519` | under 4 s |

The training runner did not measure peak host memory; it is `UNKNOWN`. Feature
extraction used the local RTX 3080 Laptop, float32, peak CUDA allocation
315,733,504 bytes, and took 2,556.99 seconds. The run receipt records Python
3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2 and PyAV 18.1.0. No cloud
compute or cloud cost was used.

## Interpretation and next gate

The joint configuration clearly outperforms uniform and the fit-derived
position prior. Against the much smaller video-only model it also improves the
fixed development metrics, but that comparison confounds capacity. The
parameter-matched video-only arm narrows the gap: joint has +2.55 points
Accuracy, whose video-cluster interval includes zero, and lower NLL/Brier/ECE.
This is suggestive evidence for the joint configuration on this task, not proof
that audio alone caused the difference or that the scorer generalizes beyond
this source/task. Accuracy remains far from the product target, candidate
counts are unequal, and the sharp train/selection divergence shows overfitting.
No 95–98% claim or product promotion is justified.

The task still lacks truth-bearing missing/shuffled/stale/timing-offset
counterfactual controls and a new independent validation cohort. Do not sweep
seeds or alter this model from the observed development scores. The next gate
should use new, video-disjoint source records for a prospective input-ablation
comparison. Keep the broader four-modality and ≤200M pre-quantization goals
unchanged.

The fixed E0G1 GPU sensitivity diagnostic was not repeated in this work: later
authorized attempts 11, 14 and 15 already passed all 44,375 forwards on the same
checkpoint and exactly reproduced prediction/sensitivity hashes. Attempt 09
remains an earlier incomplete record; the later completed evidence is in
[`R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md`](R2_AV_REPLACEMENT_DIAGNOSTIC_20261018.md).
