# LibriSpeech hard-negative transfer diagnostic

## Purpose

The Speech Commands Audio+Text readout scored highly on its fixed four-query
development task. This read-only diagnostic checks whether the same frozen
checkpoint transfers to a four-choice transcript task when the wrong answers
are random or deliberately similar in phonemes, vocabulary, and length. It does
not retrain, calibrate, or select a model, and does not load a test or sealed
audit set.

This is an out-of-distribution transfer diagnostic, not a general speech
recognition score. The trained readout saw Speech Commands keywords and fixed
question templates; this evaluation asks it to match read-audiobook utterances
to full transcripts.

## Frozen model and source

- Runtime bundle: `path-a-audio-multiquery-runtime-v0`.
- Readout SHA-256: `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.
- Audio encoder: `openai/whisper-tiny`, revision
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
- Text encoder: `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`.
- Bundle manifest SHA-256:
  `574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`.
- Baseline run-report SHA-256:
  `833258cbc584cbe5f2583c7cc9a5a6f115deb9a46b232e81b3eb7c7489fc081b`.
- Dataset: OpenSLR LibriSpeech SLR12 `dev-clean`, [official source](https://openslr.org/12/).
  The downloaded archive matched the published MD5
  `42e2234ba48799c1f50f24a7926300a1`; SHA-256 is
  `76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3`.
  OpenSLR states CC BY 4.0; retain the corpus citation and attribution.
- Pronunciations: CMUdict at
  [`74790861f652b15e4ac49015a90074ad62a27690`](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690),
  dictionary SHA-256
  `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`.
  The retained license file SHA-256 is
  `bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a`.

Only the public `dev-clean` development split was used. Of 2,703 utterances,
2,011 were fully covered by the pinned dictionary. Twenty-three more were
excluded because an option exceeded the unchanged runtime's 64-token input
limit. After requiring three valid distractors in both conditions, 1,938
targets remained: 1,938 unique audio files, 1,937 normalized target texts,
40 target speakers, and 97 chapters. Fifty otherwise eligible targets lacked
three qualifying hard distractors. No model training, test split, or sealed
audit access occurred.

## Distractor construction

Every target uses the same question, correct transcript, audio, four choices,
and seeded correct-option position in both conditions. Random distractors are
sampled from other speakers and chapters and excluded from that target's hard
candidate set. Hard distractors come from other speakers and chapters, differ
by at most one word, share at least one lexical word, and have normalized
character length ratio of at least 0.65. The top three are ranked by
CMUdict-phoneme sequence similarity, vocabulary Jaccard, then length
similarity. All four options are unique.

The realized hardness check confirms separation between conditions:

| Mean distractor property | Random | Hard |
| --- | ---: | ---: |
| Phoneme sequence similarity | 0.2306 | 0.4055 |
| Phoneme bigram Dice | 0.2261 | 0.3423 |
| Vocabulary Jaccard | 0.0426 | 0.0999 |
| Word-count difference | 11.21 | 0.65 |
| Character-length ratio | 0.5520 | 0.8895 |

## Results

The same frozen bundle was run on all 1,938 examples in each condition.
Probabilities are the runtime's unmodified softmax outputs.

| Distractors | Accuracy | NLL | Brier | ECE (15 bins) |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.2652 | 2.6085 | 1.0880 | 0.4487 |
| Hard | 0.2637 | 2.5361 | 1.0841 | 0.4425 |
| Hard minus random, paired speaker bootstrap mean | -0.0016 | -0.0735 | -0.0042 | -0.0069 |
| 95% percentile interval | [-0.0228, 0.0196] | [-0.1528, 0.0052] | [-0.0320, 0.0236] | [-0.0331, 0.0192] |

All four intervals include zero. The bootstrap resampled 40 target-speaker
groups 10,000 times with seed 17; this is explicitly low power. Hard negatives
were substantially closer by the construction checks, but they did not cause
a measurable accuracy decrease. Both conditions were near four-choice chance
accuracy and had poor out-of-distribution probability quality. This does not
show that the hard set is easier; the fixed model transfers poorly to both
forms of this transcript task.

For context only, the model's original Speech Commands validation metrics were
Accuracy/NLL/Brier/ECE-15 `0.9785/0.0693/0.0323/0.0143`. The tasks and
distributions differ, so these values are not a direct performance delta. The
result supports keeping the keyword benchmark's claim narrow and testing audio
representations on task-matched evidence before generalizing.

## Execution and reproducibility

- Source commit before this diagnostic: `e160ef877cd2b7f1b096573617ed196bb27a7f84`.
- Runner: `scripts/evaluate_librispeech_audio_hard_negatives.py`, SHA-256
  `4c6b3dd89a9427f0dfa17f3840ba160349729f964ad230b604d9217ba7895fd0`.
- Hard-negative helper: `src/tiny_omni_decision/audio_hard_negatives.py`,
  SHA-256 `06608341c72f43f97748d48107e393967e5c9d7490d7b09cb96ad191c7c4c201`.
- Ordered target-ID SHA-256:
  `709687d926f46907a3c5d447d8921b51cb8f1188163f25fd777b11dfae34384c`.
- Example-manifest SHA-256:
  `1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923`.
- Result report at
  `C:/CodexArtifacts/pretrained-reuse-r2/librispeech-audio-hard-negatives-v0/report.json`,
  SHA-256 `087988dcfc2373a8cb38f339dd26e8badd2d7e69bf76ea13ac91edcc9f56c791`.
- Random predictions SHA-256:
  `a686ba000574108242534e0fba81aac4829a3ce3e6df7c10d93d07fc2571fdf3`.
- Hard predictions SHA-256:
  `401d033949235c02cd1bc054a1db0214d1e120dea2b1298150433cff1aaff7ff`.
- Hardware: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB; Python 3.11.9,
  PyTorch 2.6.0+cu124, Transformers 5.19.0, soundfile 0.14.0, NumPy 2.4.6.
- Model load: 0.59 s; paired inference over 1,938 audio observations: 18.40 s;
  total preflight plus inference: 102.70 s. Peak CUDA allocation was 381 MB;
  cloud cost was $0.
- CPU helper tests: `4 passed`. Training performed: no. Model weights changed:
  no. Baseline checkpoint modified: no.

The run used the following PowerShell command (paths are the external,
hash-verified inputs and a new output directory):

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_librispeech_audio_hard_negatives.py `
  --dataset-root C:\CodexArtifacts\pretrained-reuse-r2\librispeech-dev-clean-hardneg-v0 `
  --cmudict C:\CodexArtifacts\pretrained-reuse-r2\librispeech-dev-clean-hardneg-v0\cmudict.dict `
  --cmudict-license C:\CodexArtifacts\pretrained-reuse-r2\librispeech-dev-clean-hardneg-v0\cmudict-LICENSE `
  --bundle C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-multiquery-runtime-bundle-v2 `
  --baseline-run-report C:\CodexArtifacts\pretrained-reuse-r2\audio-multiquery-v0\run-report.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\librispeech-audio-hard-negatives-v0 `
  --device cuda --seed 17 --audio-batch-size 8 --bootstrap-replicates 10000
```

## Limits and next use

This is public development data, not a blind or sealed audit. It is English
read-audiobook speech, not conversational speech. CMUdict coverage excludes
some rows, and phoneme-sequence matching plus lexical/length constraints are
deterministic proxies rather than human judgments of ambiguity. The same
development transcript pool supplies distractors; the speaker bootstrap groups
target clips but does not model every dependency induced by reusing candidate
transcripts. Treat intervals as a limited diagnostic, not a formal population
confidence guarantee.

This result does not establish synchronized audio-video understanding. R2
still lacks an approved task-matched joint audio-video source, and the separate
runtime bundles exceed the provisional package-size target. Do not promote the
current audio readout or broaden its quality claim based on this experiment.
