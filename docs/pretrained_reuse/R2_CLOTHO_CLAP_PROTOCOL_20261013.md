# Fixed CLAP audio representation comparison

Protocol fixed before inspecting this comparison's validation predictions.
This is a bounded Path A representation experiment under the existing autonomous
local-work authorization. It does not qualify a complete Omni product.

## Question and execution gates

Does frozen CLAP audio provide better conditional decisions than the measured
EfficientAT audio representation on the identical frozen Clotho-AQA development
questions? Audio-only remains a criterion-removal control, not a general audio score.

Use the pinned, strictly loaded 28,190,872-parameter audio branch and official
48 kHz frontend. Cover each complete clip in consecutive <=10-second windows;
repeat-pad only the final partial window. Require exact same-device replay and
full sample coverage. Duration-weighted normalized pooling is fixed and discards
order; retain window vectors/timestamps separately. Do not randomly truncate.

CPU/CUDA16-clip smoke: full coverage, exact repeats on each device; CPU/GPU
max pooled difference3.65e-7, window6.31e-7 (tolerance1e-5).
CPU9.08s/CUDA5.79s, 1.57x for this smoke only. CUDA allocator peaks178.6MB
allocated/192.9MB reserved exclude driver/context memory. Full attempt01 was
incomplete at813 clips: Windows atomic progress replacement raised PermissionError;
812 clips were in the last atomic report. Process terminated. Preserve attempt01,
test bounded save retry and run fresh extraction in attempt02, <=900s.

## Capacity and data control

Keep original train2312/calibration550/validation560 questions and all source,
uploader, labels, IDs, order and integrity gates. Fit deterministic sign-canonical
PCA512→384 on **578 unique training assets only**, with no labels or held-out
assets. Save the mean, components, fit IDs, variance and hashes. Fail closed on
insufficient rank or incomplete/corrupt extraction. PCA is not tuned on validation.

Retain the original frozen MiniLM question features. Train the exact same
49,409-parameter candidate scorer in audio-only and audio+question conditions:
seed17,24epochs,batch128,AdamW lr.001/decay.001,770→64→1, same sampler,
train-question-weighted normalization, earliest minimum raw validation NLL,
calibration-only17-temperature grid(.25–4). Separate best/final checkpoints and
new output directories. Reuse the prior question-only condition without refitting.
No extra seeds, epochs, hidden-width search or encoder fine-tuning.

Projected runtime coefficients197,120 (512 mean +512x384 matrix) count toward
the budget even though frozen. Audio+question+scorer+projection=45,827,225
resident parameters/coefficients; this excludes other Omni modalities. Audio
representation, frontend, pooling and compression form one replacement bundle;
any gain does not identify which of those elements caused it.

## Evaluation and decision

Report Accuracy/NLL/Brier/ECE, fixed train/validation gaps and all24 points.
Replay saved weights and predictions. Use the same frozen source-deranged and
train-mean audio controls, with fixed selected weights/temperature. Targets remain
those of original pairs; controls diagnose incorrect evidence, not replacement labels.

Compare joint against old EfficientAT joint and unchanged question-only using
paired uploader-cluster bootstrap (81 validation uploaders,10,000 draws,seed17,
question-weighted differences,percentile95%). Prefer NLL/Brier improvements
without Accuracy harm, alongside useful matched-audio contribution. Mixed or
negative results remain recorded and do not justify product adoption or more epochs.
Intervals are descriptive: validation already selects checkpoints and is reused.

CLAP upstream checkpoint selection used Clotho retrieval. HF checkpoint ancestry
and pretraining-record overlap are UNKNOWN. These results cannot be presented
as a new blind benchmark. No official AQA validation/test or sealed audit is opened.
Local compute only; cloud cost0. Existing experiments remain unchanged.
