# R2 LibriSpeech Whisper Tiny option-likelihood diagnostic — 2026-11-05

## Scope and result

This is a one-time, frozen-model comparison on the existing fixed LibriSpeech
dev-clean four-choice transcript task. It compares the existing Path A
Speech-Commands Audio+Text readout with a pinned Whisper Tiny model, scoring
each supplied transcript by teacher-forced mean log likelihood conditioned on
the audio. Both return normalized probabilities over the exact same supplied
options. No model was trained, calibrated, or selected; the existing baseline,
examples, audio archive and model weights were read-only. The official test and
project sealed audit were not loaded.

Whisper Tiny selected the correct transcript for 1,937/1,938 examples in both
the random-distractor and hard-distractor conditions (99.948% accuracy; one
error in each condition). The existing frozen Path A readout scored 26.52% and
26.37% on those same rows and options. This is strong evidence for the narrow
task of matching English audiobook speech to one of four candidate transcripts.
It is not evidence of general audio reasoning, arbitrary text-conditioned
decision quality, synchronized multimodal fusion, or product quality. Whisper
pretraining overlap with LibriSpeech is unknown and could explain some or all
of this result; the dev-clean set is previously inspected development data.

## Frozen inputs and scoring

| Input | Pinned identity | SHA-256 |
|---|---|---|
| Model | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; Apache-2.0 per pinned card | `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395` |
| Model size | 37,760,640 parameters; checkpoint 151,061,672 bytes | — |
| LibriSpeech dev-clean archive | OpenSLR SLR12; CC BY 4.0 | `76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3` |
| Fixed examples/options | 1,938 ordered examples; random and hard options; target positions unchanged | `1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923` |
| Ordered sample IDs | Hash definition matches the original diagnostic's newline join with no trailing newline | `709687d926f46907a3c5d447d8921b51cb8f1188163f25fd777b11dfae34384c` |
| Existing Path A baseline report | Unchanged frozen Speech Commands Audio+Text readout | `087988dcfc2373a8cb38f339dd26e8badd2d7e69bf76ea13ac91edcc9f56c791` |

The evaluator hash-checks the archive, fixed manifest, model checkpoint and
baseline report before loading. It decodes FLAC from a sequential tar stream
without extraction and verifies every referenced media SHA-256 and duration.
Candidate strings are used verbatim after stripping outer whitespace. The
Whisper English transcription prefix is fixed to start-of-transcript, English,
transcribe and no-timestamps. Probabilities are a four-way softmax over mean
log probabilities for transcript tokens plus EOS; no temperature or fitted
calibration is used. Option reversal changes probabilities by at most
`1.79e-7`.

## Results

| Fixed condition / model | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Random distractors — Whisper Tiny | 0.999484 | 0.108265 | 0.019961 | 0.099430 | 0.900449 |
| Hard distractors — Whisper Tiny | 0.999484 | 0.107728 | 0.017809 | 0.100229 | 0.899651 |
| Random distractors — unchanged Path A readout | 0.2652 | 2.6085 | 1.0880 | 0.4487 | not recorded in baseline summary |
| Hard distractors — unchanged Path A readout | 0.2637 | 2.5361 | 1.0841 | 0.4425 | not recorded in baseline summary |

The 10,000-replicate paired speaker bootstrap (40 speakers, seed 17) compares
hard minus random on the same examples. Accuracy difference is 0.0 with a
degenerate `[0, 0]` interval because both conditions have the same accuracy.
NLL difference is -0.000489 (95% percentile interval `[-0.004006, 0.003023]`);
Brier difference is -0.002138 (`[-0.003286, -0.000954]`); ECE-15 difference is
0.000842 (`[-0.002252, 0.003938]`). The bootstrap gives no evidence that the
hard condition changes accuracy or NLL; the small Brier difference is specific
to this exposed set and this scoring rule.

The strong transcript match must be interpreted with the training-overlap
limitation. Since Whisper is an ASR model pretrained on a very large audio
corpus, this evaluation does not establish decontaminated generalization. The
task also supplies the exact transcript as one choice, making this a useful
ASR-derived candidate scorer rather than an open-ended reasoning test. The
result may justify considering a Whisper likelihood route for this specific
task family; it does not justify replacing the shared Path A architecture or
claiming four-modality quality.

## Execution and artifacts

- Source commit: `2931f354045efca44c1ddcfe093d58b83bc26cae`.
- Pre-registration config: `configs/pretrained_reuse/librispeech_whisper_option_likelihood_v0.json`, SHA-256 `d439942f6d770f7f1b9643be59188b88efc201edbd4de99bed34bc1b01fe7933`.
- Runner: `scripts/evaluate_whisper_librispeech_candidate_likelihood.py`, SHA-256 `6cb83c555e216ed04445ab40e974d37ecc7a02fa90a9df0a860572c4bfeb0128`.
- Scoring helper: `src/tiny_omni_decision/whisper_option_likelihood.py`, SHA-256 `480da326ca804eceb479f04af4daa0cc0d22d49903e22d5ef58a08dec4799420`.
- Focused CPU tests: 9 passed; `ruff check` and `ruff format --check` passed for the changed code.
- Full local suite with the optional ML dependencies already installed: 2,937 passed / 11 skipped. A second full-suite run with `torch` import-blocked to match CI's `.[dev]` dependency set passed 2,197 / skipped 183; the new focused module correctly skips when PyTorch is absent.
- Remote CI on code commit `213a5ed02d2c4431fa0179a78fe2c0d8c315031e`: push run `41876492567` and PR run `41876508803` both passed. An earlier CI failure exposed the test's unconditional optional `torch` import; it was corrected without adding PyTorch to the lightweight dev dependencies.
- Smoke: 8 fixed examples completed in 5.00 seconds on CPU; no GPU used.
- Full run: 1,938 examples × two conditions; 20m59s end-to-end, including model load and 10,000 bootstrap replicates. Candidate scoring took 1,230.06s; no GPU or cloud compute was used.
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0; FP32 CPU inference, two CPU threads.
- Run report: `C:\CodexArtifacts\pretrained-reuse-r2\librispeech-whisper-tiny-option-likelihood-20261105-attempt-01\report.json`; SHA-256 `1bec60cf15e3c7bfe27a6cca3202b88d791c6b4a33112e6556b90d402cdfd856`.
- Per-example predictions: same directory, `predictions.jsonl`; 3,876 rows (1,938 × two conditions), SHA-256 `61d4d82a5e56f89185e848a2e9f5b0cc55d375ce65bf29812cb99dad3d4c671e`.
- Configured model checkpoint size is 151,061,672 bytes. A complete deployable package/runtime size was not measured, and no quantization was performed.

## Limits and next gate

Keep this result as narrow ASR-style candidate-likelihood evidence. Do not
retune on these examples, call it blind, or infer product-level quality from
the 99.948% figure. The most useful next gate is a genuinely independent,
licensed speech transcript set that was not used to train or select Whisper,
plus an end-to-end Path A runtime/package measurement if this route is later
considered for the product. No code here is wired into the product dispatcher,
and R2-M, R7 packaging and the 90–98% four-modality goal remain open.
