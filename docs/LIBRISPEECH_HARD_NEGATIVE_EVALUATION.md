# LibriSpeech Hard-Negative Audio Evaluation

## Purpose and protocol

Audio achieved 100% accuracy on the original 512-example clean-dev-v2 selector. To check how sensitive that score is to distractor choice, this evaluation retains the exact same 512 audio records, target transcripts, audio files, four-choice count, and question. It replaces only the three wrong answer choices.

For each selector record, the script ranks other LibriSpeech clean-dev-v2 validation transcripts after excluding all 512 selector IDs. It chooses the three highest-scoring distinct transcripts using this deterministic similarity proxy:

- 50% Soundex sequence similarity (English phonetic proxy)
- 30% word-set Jaccard overlap
- 20% normalized character-length ratio

The choice order is shuffled deterministically from the sample ID. Each generated row preserves the original audio reference and correct transcript. The original predictions and corpus remain untouched.

Both conditions use the same selected Teacher adapter at step 768, SHA-256 `bb6c429d0cfc887e75a2a4142a699db8716a4f6ea5af84e9d970390c1b6cd33b`. The original condition uses its saved step-768 predictions; the hard-negative condition was evaluated against the retained step-768 best adapter inside the step-896 resume snapshot. The later step-1024 `best/` adapter was deliberately not used.

## Results

The first pass below is a preliminary comparison: it reused the original
step-768 baseline predictions (configured input limit 1,024) and evaluated the
hard-negative choices with an effective limit of 1,223. The checkpoint and
512 audio IDs match, but the input-length setting did not. Do not interpret the
small metric delta as a fully controlled distractor-only comparison.

| Distractors | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Original validation choices | 1.0000 | 0.000464 | 0.000019 | 0.000457 | 0.999543 |
| Similarity-ranked hard choices | 0.9961 | 0.005891 | 0.003172 | 0.004273 | 0.996005 |
| Hard minus original | -0.0039 | +0.005427 | +0.003153 | +0.003816 | -0.003538 |

The hard set changed two of 512 predictions from correct to incorrect. Its average target-to-distractor similarity score was 0.369 versus 0.157 for the original wrong choices. In all 512 rows, the best-ranked hard distractor scored above the best original distractor under the same proxy.

## Interpretation and limits

The preliminary pass suggests that the original LibriSpeech score is insensitive to this particular distractor change: accuracy remains 99.6%. Because the input limits differed, this is not yet a controlled estimate of distractor difficulty. It also does not establish general audio competence or performance on other speakers, noise, accents, tasks, or recording conditions.

Soundex is only a deterministic spelling-to-sound proxy. It is not a pronunciation lexicon, phoneme-level distance, human ambiguity judgment, or an adversarially validated benchmark. Its average score remains moderate, so this should be called a similarity-ranked distractor check rather than a definitive phonetic challenge set. A stronger follow-up would use a pinned pronunciation lexicon or ASR-derived phoneme sequences and human review, while keeping source/audio identities and splits fixed.

This is development validation reused to compare distractor protocols, not a new blind audit. It was not used to select a training checkpoint or change training. The learning run's config, training data, checkpoints, and selected step were not modified. The preliminary hard-negative pass raised its prompt limit from 1,024 to 1,223 tokens because replacement transcripts lengthened some prompts.

## Matched paired evaluation

A separate evaluation output is stored at
`artifacts/teacher-quality-next/evaluations/librispeech-hard-negative-matched-step768-v2/`.
It contains the same 512 examples and hard-negative counterparts paired by ID.
The evaluator loaded the pinned step-768 adapter once and measured the maximum
processor input length across both conditions, then re-inferred both at the same
effective limit of 1,223. Historical predictions at 1,024 remain unchanged.

| Distractors | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Original, matched re-inference | 1.0000 | 0.000464 | 0.000019 | 0.000457 | 0.999543 |
| Similarity-ranked hard negatives | 0.9961 | 0.005891 | 0.003172 | 0.004273 | 0.996005 |
| Hard minus original | -0.0039 | +0.005427 | +0.003153 | +0.003816 | -0.003538 |

The same adapter (SHA-256
`bb6c429d0cfc887e75a2a4142a699db8716a4f6ea5af84e9d970390c1b6cd33b`), IDs,
audio, target, question and option count were used; only distractors changed.
Two of 512 predictions changed from correct to incorrect. This shows the
generated set was modestly harder under this proxy, but the score remained
99.6%. It does not establish robust phoneme-level challenge or broad speech
understanding. The set is development validation, not a blind audit, and did
not select a checkpoint or change training.

The independent learning run later reached its declared 2,048 updates but
failed while finalizing its result because `corpus-manifest.json` was absent.
That post-run metadata failure is detailed in
`docs/VIDEO_TEACHER_V2_LONG_BUDGET.md`; it did not arise from this paired
evaluation. Inference wrote only to the separate evaluation directory and did
not modify training configuration, corpus, checkpoint, optimizer, sampler or
run state.

## Reproduction artifacts

- Generator and evaluator: `scripts/evaluate_librispeech_hard_negatives.py`
- Similarity and selection logic: `src/tiny_omni_decision/audio_evaluation.py`
- CPU tests: `tests/test_audio_evaluation.py`
- Local artifacts: `artifacts/teacher-quality-next/evaluations/librispeech-hard-negative-step768-v1/`
- Matched evaluation metrics: `artifacts/teacher-quality-next/evaluations/librispeech-hard-negative-matched-step768-v2/hard-negative-metrics.json`
- Hard corpus SHA-256: `f87791b106c8fd5135a84f672fb0b96259b4d2fec1af23f809af238e47d86e14`
- Metrics SHA-256: `8cc45c37590abbbf4afa7b23e2fab6f08328ccf9919d423006864e938f74bdb9`
- Predictions SHA-256: `570d3c9d4c82d1479df84f153ac4aa62cfb91b2ce7f5f8ff30ff86b0f8836449`

Matched metrics SHA-256: `1a96e1b1eb7a0978dcc355063bba423c58fa1d32c5b17145580672d0b1c171cf`;
matched hard predictions SHA-256:
`570d3c9d4c82d1479df84f153ac4aa62cfb91b2ce7f5f8ff30ff86b0f8836449`;
matched original predictions SHA-256:
`2627528adf65f97577e07a4c80bebac4e3164ab743c75ed043acf4fe3bf89365`.

The generated corpus, per-row distractor audit, metrics, and predictions are local artifacts and are not committed to Git.

To prepare a new output directory and evaluate it without replacing this result:

```powershell
python scripts/evaluate_librispeech_hard_negatives.py --prepare-only --output artifacts/teacher-quality-next/evaluations/librispeech-hard-negative-reproduction
python scripts/evaluate_librispeech_hard_negatives.py --evaluate-only --output artifacts/teacher-quality-next/evaluations/librispeech-hard-negative-reproduction
```
