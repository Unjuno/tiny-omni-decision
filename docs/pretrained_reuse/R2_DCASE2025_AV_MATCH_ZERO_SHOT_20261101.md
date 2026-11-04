# DCASE2025 AV match — direct frozen-feature cosine control

## Scope

This follow-up tests whether the frozen pretrained audio and video features
already share a directly usable correspondence space. It adds no learned
projection, calibration, labels-to-parameters path, checkpoint, or training.
It uses the exact same 256-choice DCASE dev-train task and candidate order as
the nested-room CV result in
[`R2_DCASE2025_AV_MATCH_CV_20261101.md`](R2_DCASE2025_AV_MATCH_CV_20261101.md).
The public dev-test annotations and sealed audit were not accessed.

Each feature vector is independently L2-normalized; raw cosine similarity of
query audio against each of four candidate video vectors is softmaxed at fixed
temperature 1. The baseline, task rows, feature arrays, and saved OOF reference
predictions are hash-checked before evaluation. The trained-CV metrics below
are recomputed from its already saved OOF probabilities; no model is reloaded.

## Reproducibility

- Source commit before the follow-up: `4af57ac80fd1bd11ce0d47ced2c6e601e1bb9089`.
- Executed script SHA-256 (copied beside the output):
  `ab3fe14f63354cfa168cb42846039cd0702234cb58a887554fcd440a5fe42b36`.
- Checked-in script SHA-256 after lint-only string wrapping:
  `e28cbd78d23e42e79a1187c3e5f937fefbd198d962008b3f400a92872186db8d`.
- Effective config SHA-256:
  `0b5ad53931189bc608ecfe473f55afa1fa226e5b9f36583aedc972f2042ac283`.
- Result JSON SHA-256:
  `95dfa5c49151b12c820adf0c5a147630d32aaaa46553defc653350a0c9d472d6`.
- Prediction JSONL SHA-256:
  `58f518619dce013515f94a8554be40182e409023adac83e119c5e75463fe2c9c`.
- Output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-match-zero-shot-20261101-attempt-01`.
- Inputs: task manifest SHA-256
  `017b5a915396f52e8e2a07b35dfcefa82071df158d0df0a6bb1b6af7d57a71ce`;
  audio feature SHA-256
  `5025bbc4f718827d1afd779fdacd9e77253f4d2f22416866e765ad9652ec9618`;
  video feature SHA-256
  `5c04393fcef5f501ce77c10775000c0b88729b015e457927e32d2b110e7c20aa`;
  prior CV predictions SHA-256
  `f49cec00e59ce58749a69fb1dba0ea0a1e5dee9e131a738e24d0f3eacea9691c`.
- Python 3.11.9, NumPy 2.4.6, Windows 10 build 26300, Intel i7-12700H.
  CPU only; no GPU, cloud, media access, or model-weight load. Elapsed time
  was not captured by this runner and is UNKNOWN.

The executed command was:

```powershell
python scripts/evaluate_dcase2025_av_match_zero_shot.py --features C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-features-20261031-attempt-01 --task C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-matching-task-20261101-attempt-01 --reference-run C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-match-cv-20261101-attempt-01 --output C:\CodexArtifacts\pretrained-reuse-r2\dcase2025-av-match-zero-shot-20261101-attempt-01 --replicates 10000 --seed 17
```

## Results

There are 256 queries, 78 parent recordings, and 9 rooms. Four-choice chance
is 25% (64 correct). ECE uses 15 bins.

| Arm | Accuracy | Correct | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Direct frozen cosine | 27.34% | 70 | 1.3861 | 0.7499 | 0.0226 | 0.2508 |
| Uniform | 25.00% | 64 | 1.3863 | 0.7500 | 0.0000 | 0.2500 |
| Trained nested-room CV | 26.56% | 68 | 1.4104 | 0.7615 | 0.0744 | 0.3264 |

Paired cluster bootstrap used 10,000 resamples, seed 17, with parent-recording
and room clusters sampled separately. Direct-cosine minus uniform Accuracy
was +2.34 percentage points; the parent interval was [−5.19, +10.08] points
and the room interval [−2.07, +6.59]. Parent NLL and Brier intervals also
included zero. Room-cluster NLL difference was −0.000159 (95% CI
[−0.000293, −0.000011]); Brier difference was −0.000079 (95% CI
[−0.000146, −0.000006]). Only nine rooms support that room-level interval.

Direct cosine minus trained-CV Accuracy was +0.78 points (parent 95% CI
[−6.20, +7.69], room CI [−4.85, +5.54]). NLL difference was −0.0242 and
Brier difference −0.0116; their parent intervals included zero, while the
nine-room intervals were respectively [−0.0432, −0.0033] and
[−0.0219, −0.0008]. These are paired post-hoc development comparisons on the
same exposed source task and are not independent confirmation.

## Interpretation

The untrained features contain at most a weak direct AV correspondence signal
on this task: six more correct predictions than chance, with wide parent-level
uncertainty. Direct cosine has lower confidence and better NLL/Brier than the
small trained readout, but does not produce reliably higher Accuracy. The
trained head's positive train/validation gaps remain consistent with
overfitting; a more complex shared Lens is not justified by this result.

Keep the existing nested-CV and direct-cosine artifacts as development
references. Do not tune on this task or promote the feature pair. Next R2 work
needs a separately sourced, sufficiently grouped task with exact temporal
correspondence and clear both-input necessity; the current DCASE match probes
do not meet that quality gate.
