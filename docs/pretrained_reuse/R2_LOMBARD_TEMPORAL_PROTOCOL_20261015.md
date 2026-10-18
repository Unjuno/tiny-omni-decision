# Native Lombard command readout: fixed temporal-information comparison

Goal: test whether retaining already extracted chronological visual information
improves native six-slot command decisions or useful audio/video fusion. This
is a scoped Path A experiment; it cannot establish generic Omni95–98% quality.
No signal degradation, new frames, encoder updates, labels or source changes.

Retain all4080train/398calibration/889development-validation utterances, all51
options including25letters, current grouped source/content splits and all existing
source/media/model/cache hash gates. Development has already been observed; it
is not blind audit. Never load sealed data. The previous pooled24epoch results
remain immutable and are context, not a capacity-matched causal comparison.

Use exact existing `framewise.npy`5367×16×768 FP32 native VideoPrism vectors,
SHA `a4f44cd94ef762316440e359f5e916a0e862b2e61378aeafa0bd6f4e9f67699a`;
existing pooled16-frame mean and complete physical Whisper-token mean384D.
Require full ordered identities, no labels/filenames/alignments as neural input,
and all frame means agreeing with pinned pooling. Normalize with the same
training-only pooled feature mean/std in both conditions, preserving originals.

One fixed head: visual Linear768->64/GELU, unidirectional1layer GRU64->64,
last visual state concatenated with384D audio and2presence bits, then
Linear450->64/GELU/Linear64->51.106,355 registered/trainable parameters. Absent
visual final state is explicitly zero; absent audio is zero with mask clear.
No text, transcript, decoder or mandatory condition encoder is loaded.

Five fixed arms: audio, video_broadcast, video_ordered, joint_broadcast,
joint_ordered. Broadcast repeats each unchanged pooled visual vector16 times;
ordered uses real chronological frame vectors. Every arm has identical head,
seed17initial weights, sample order, batch128, AdamW fixed LR0.001/WD0.001,
24epochs/768updates, six equally weighted slot CE. Thus ordered/broadcast tests
temporal information at equal capacity and training budget, although changing
time variation as well as order. A reversal control helps distinguish order
dependence from mere variability. Do not call this a pure order-only effect.

Select earliest minimum unrounded raw validation macro NLL; retain best/final
and optimizer/RNG state separately. Calibrate selected weights on independent
398calibration examples using unchanged17temperatures0.25..4 per slot.
Log all24train/validation curves, consumed versus unique versus repeated counts,
actual selected-step counts, loss/gradient norms/time and per-slot/macro/minimum
Accuracy/NLL/Brier/ECE plus all-six-correct and mean confidence. Preserve per-row
IDs/targets/raw logits/probabilities at every validation, best/final and controls.

All selected joint controls use frozen weights/normalization/temperatures:
matched, chronological reversal, deterministic different-speaker video mismatch,
audio mismatch, each modality absent and native option permutation. Evaluate
best/final checkpoint byte parity after reload. Paired10000-resample speaker-group
bootstrap(seed17, percentile95%) on primary ordered-minus-broadcast comparisons
and joint-minus-audio. Report development-selection and only9speaker limitations;
do not claim blind confidence, natural-noise robustness or oracle decision rules.

Before full run, separate train-only four-utterance/two-epoch smoke checks
dimensions/gradients/finite numerics, checkpoint/JSON parity, frozen source pins
and projected full runtime. CPU4threads/FP32/deterministic, explicitly CPU-only:
new readout training runs on immutable cached features; no GPU/encoder training.
Local body7200s/peak3GiB/new output128MiB/disk reserve1GiB. Existing GPU8GiB guard
and interrupted run remain untouched. Stop on integrity/NaN/Inf/resource failure;
quality decline alone does not stop fixed24epochs. No extra seed/budget/grid.

Execution staging: this producer revision admits smoke only. Its reusable
training functions do not admit full CLI execution. A separate reviewed full
controller must bind verified smoke receipts, fixed controls/bootstrap and a
fresh-process evaluator before any full fit begins. Smoke success alone is
neither the comparison result nor completion of the product goal.

All simultaneously resident frozen encoder parameters plus full head and
normalization coefficients are reported; no complete4modality/package claim.
Existing speech/video encoders total122,573,568 parameters, joint total with
head and2304normalization coefficients122,682,227. Encoders are not loaded during
the cache-only fit. Their deployment weight costs are still part of composition.
Publish negative results and all attempts. No model promotion/main merge,
quantization, Recovery, paid compute, legacy checkpoint or corpus overwrite.
