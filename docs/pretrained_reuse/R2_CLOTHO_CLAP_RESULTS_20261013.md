# CLAP representation comparison: completed, not adopted

Protocol: [fixed comparison](R2_CLOTHO_CLAP_PROTOCOL_20261013.md).
Learning source commit `580ff667b3257dcac2ad96f0a9d1f161a4706023`.
All existing Clotho/EfficientAT/model artifacts are retained. No official AQA
validation/test, sealed audit, new seed, encoder training or paid compute was used.

## Actual execution and integrity

Attempt01 terminated during atomic Windows progress replacement: last atomic
report812/856 clips; temporary FAIL report813 clips,199.79s,PermissionError.
The exact lock holder is UNKNOWN. A bounded1-second permission retry was tested,
with persistent failures still raised; original attempt preserved. Fresh attempt02
completed856 clips/2,270 windows/925,014,011 samples in202.64s CUDA.
All samples covered exactly once, finite features and exact repeated-window
outputs. The16-clip CUDA smoke and full extraction pooled features match bitwise.
CPU/GPU smoke differences remain within1e-5, not bitwise across devices.

Full extraction used frozen FP32 CLAP audio, strict pinned weights/config/frontend,
seed17,deterministic kernels,TF32 off. RTX3080Laptop16GB allocator peaks178,565,632
bytes allocated/192,937,984 reserved; these exclude context/driver memory.

Train-only PCA uses578 unique training audio assets, no labels or calibration/
validation assets.384 components retain99.992284% variance. Its197,120 runtime
coefficients count toward the45,827,225 audio+question+head+projection total.
This is a two-input research runtime, not the complete Omni model. Independent
preflight projection and loaded projection reproduce features bitwise.

FeatureSHA `b965887b0fd1505af06bcbad6b3c0fcbd808cae31f31559636328543f27722fe`.
WindowSHA `d716cce34718f4aea044dad96d8b1681ecb5eb4ea559424a86e1d96871e1ce34`.
Effective protocolSHA `15717f6bc17f57a1904fb6d3c65a6d7745e871cc66f692aa1f31cd62eec91a46`.
Frozen corpus hashes and all encoder/projection/artifact hashes are in the adjacent
JSON and local protocol. Existing split/source/uploader/media gates passed.

## Same-data selected comparison

All learned conditions retain the49,409-parameter scorer,seed17,24epochs,
AdamW.001/decay.001,batch128,identical query order,train-only normalization,
earliest minimum raw validation NLL and calibration-only fixed temperature grid.
Each new arm consumes55,488 presentations of2,312 unique questions/578 sources;
selected epochs consume6,936 audio-control and13,872 joint presentations.
Validation:560 questions/140 audio sources/81 uploader groups.

| Condition | Best epoch | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Question only, existing | 8 | .648214 | .623079 | .433142 | .028331 |
| EfficientAT+question, existing | 7 | .669643 | .624661 | .434001 | .025451 |
| CLAP audio only, question removed | 3 | .566071 | .685344 | .492163 | .024992 |
| CLAP+question | 6 | .651786 | .633886 | .442531 | .028006 |
| CLAP joint, source-deranged audio | fixed6 | .610714 | .651348 | .459496 | .032866 |
| CLAP joint, audio set to train mean | fixed6 | .637500 | .633284 | .442785 | .023351 |

The audio-only arm removes the criterion in a question-dependent task;56.61%
is not a general environmental-audio capability score. Controls keep original
targets and selected weights/temperature; they diagnose sensitivity to incorrect
evidence, not replacement-audio ground truth. All4 new selected/control prediction
vectors reproduce560 probabilities bitwise after independent checkpoint reload.

## Paired uncertainty and interpretation

Paired uploader-cluster bootstrap:81groups,10,000 draws,seed17,
question-weighted differences,percentile95%. Same reused validation selects
checkpoints; intervals are descriptive, not blind or selection-adjusted.

| CLAP joint minus comparison | Accuracy difference [95% interval] | NLL difference [95% interval] |
| --- | --- | --- |
| Question only | +.36pp [-2.87,+3.57] | +.010807 [-.003959,+.024641] |
| EfficientAT joint | -1.79pp [-5.08,+1.19] | +.009226 [-.002510,+.020829] |
| Source-deranged | +4.11pp [+.18,+7.38] | -.017462 [-.031158,-.002275] |
| Train-mean audio | +1.43pp [-2.17,+5.11] | +.000602 [-.012163,+.012483] |

The model reacts usefully to matched audio versus deliberately mismatched audio,
but improvement over question-only is not established. Compared with EfficientAT,
point estimates worsen in Accuracy/NLL/Brier/ECE and intervals include no change.
Do not claim significant degradation, superior audio reasoning, or a model-family
ceiling. CLAP upstream selection used Clotho retrieval; exact HF ancestry and
pretraining-record overlap remain UNKNOWN. These are development measurements.

## Fixed-example overfitting replay

Raw uncalibrated predictions, fixed complete train/validation examples:

| Arm/checkpoint | Train Accuracy | Val Accuracy | Train NLL | Val NLL |
| --- | ---: | ---: | ---: | ---: |
| Audio best3 | .568772 | .566071 | .674466 | .683935 |
| Audio final24 | .625865 | .530357 | .624968 | .740554 |
| Joint best6 | .793685 | .651786 | .499135 | .640641 |
| Joint final24 | .951125 | .625000 | .173893 | .941596 |

Joint gap grows from14.19 to32.61points; training NLL falls while validation NLL
rises. All48 scheduled learning points remain in the adjacent CSV. Best/final
weights are distinct. More epochs are not justified by this curve.

## Artifacts, costs and next step

- `C:/CodexArtifacts/pretrained-reuse-r2/clap-full-clip-features-01/`: preserved failure.
- `.../clap-full-clip-interruption-audit-01/`: failure/source record.
- `.../clap-full-clip-features-02/`: full features, ordered windows and executed source.
- `.../clap-full-clip-coverage-audit-01/`: coverage and smoke/full parity.
- `.../clotho-clap-preflight-01/`: separate projection/input preflight.
- `.../clotho-clap-comparison-01/`: projection, protocol, best/final head weights,
  normalization, histories, all validation IDs/probabilities and fixed audio controls.
- `.../clotho-clap-comparison-audit-01/`: reloads, fixed gaps and bootstrap.
- `.../clotho_clap_comparison_audit.py`: exact independent audit runner;
  launched with `runpy.run_path` from the worktree after direct-file launch failed
  to import `scripts`. That import failure occurred before any inference/output.

Commands: `python -m scripts.extract_clap_full_clip --scope full --device cuda
--max-seconds 900 --output C:/CodexArtifacts/pretrained-reuse-r2/clap-full-clip-features-02`;
`python -m scripts.train_clotho_clap_comparison --features
C:/CodexArtifacts/pretrained-reuse-r2/clap-full-clip-features-02 --output
C:/CodexArtifacts/pretrained-reuse-r2/clotho-clap-comparison-01`.
Torch2.6.0+cu124,Transformers5.6.2,NumPy2.4.6,PyAV18.1.0;
4CPU threads for extraction/readout, no environment upgrades.
New scorer learning/selection1.32s CPU total; no GPU backpropagation. CUDA
extraction attempts+smoke408.22s processing wall time, excluding loads; actual
GPU utilization/energy and local electricity cost UNKNOWN. Cloud cost0.

**Decision:** retain EfficientAT as this task's compact reference. Do not replace
it with the26.23M larger CLAP branch on these results or extend this run's budget.
Encoder replacement alone did not fix conditional generalization. Next investigate
whether the fixed fusion/readout preserves question–observation interactions,
and use new predeclared controls rather than repeatedly tuning this validation.
Joint audiovisual evidence and the95–98% per-task goal remain unmet.
