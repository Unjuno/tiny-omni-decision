# Japanese BERT readout probes (2026-11-23)

## Result

The Japanese BERT-base probe is worth retaining as a **text-path candidate**.
On the same fixed JCommonsenseQA development validation IDs, frozen
`tohoku-nlp/bert-base-japanese-v3` plus a 98,561-parameter option scorer
reached 70.96% Accuracy, versus 50.70% for the frozen Ruri-v3-70m reference.
The paired question-group bootstrap interval for the Accuracy difference was
+17.57 to +22.92 percentage points. NLL and Brier also improved. ECE was
slightly worse (0.0383 versus 0.0292). This is a large, useful **development
signal**, not a blind or independent test result: the same exposed validation
was used to select both checkpoints and compare them.

This does not establish a general Decision model, any image/audio/video
ability, or the 95–98% aspiration. The BERT encoder alone has 111.21M
parameters and its FP32 weight file is 447.4 MB. The combined readout has
111,305,729 pre-quantization parameters. Quantization, runtime package size,
latency, and multimodal combination were not tested.

## Japanese NLI probe

A separate check tested whether the encoder can support another Japanese
paired-text decision task. JGLUE JNLI train and publisher-dev rows were joined,
then regrouped by source YJ Captions image identity and normalized sentence
content. The frozen seed-17 split contains 18,005 train rows / 6,456 groups and
4,502 validation rows / 1,641 groups. There is zero cross-split group, image,
or normalized-sentence overlap. The publisher's official test file was not
downloaded or opened. The pinned JGLUE v1.1.0 release is CC BY-SA 4.0; its
records derive from caption/image data, so this remains within-source transfer,
not an independently sourced benchmark.

The model was frozen and jointly encoded each premise/hypothesis pair. A
98,819-parameter 128-hidden-layer classifier was selected by validation NLL
from eight readout epochs (best epoch 6). It scored:

| JNLI validation | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Frozen Japanese BERT-base + readout | 78.10% | 0.5592 | 0.3156 | 0.0199 |

The class distribution is imbalanced. Per-class accuracy was entailment
24.18% (612 rows), contradiction 78.44% (1,368 rows), and neutral 91.00%
(2,522 rows); an always-neutral baseline is 56.02%. The low entailment score is
a real weakness and prevents interpreting 78.1% as uniformly strong reasoning.
This is another development validation with checkpoint selection, not blind
generalization evidence.

## Same-task JCommonsenseQA comparison

Both candidates used the exact same question-grouped seed-17 split: 8,054 train
rows / 7,938 groups and 2,004 validation rows / 1,984 groups, with zero
cross-split normalized-question overlap. It retains the publisher's train and
dev rows after regrouping; the pinned JGLUE release has no test payload. No
sealed audit or publisher test was opened.

The BERT encoder jointly encoded each question with each of five supplied
choices. A shared 768 → 128 tanh → 1 scorer was fitted for eight epochs; best
validation NLL selected epoch 7. The Ruri baseline uses mean-pooled independent
question/choice embeddings with a `[q,c,abs(q-c),q*c]` scorer and selected epoch
7. This compares two frozen representation/readout pipelines with different
feature construction and parameter counts; it does not isolate encoder family
as the only changed variable.

| Candidate, selected checkpoint | Total pre-quant parameters | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Ruri-v3-70m + readout | 70,199,681 | 50.70% | 1.2193 | 0.6129 | 0.0292 |
| Japanese BERT-base + readout | 111,305,729 | 70.96% | 0.7904 | 0.4047 | 0.0383 |

Paired question-group bootstrap (10,000 resamples of the 1,984 groups,
seed 17; percentile 95% interval) gave candidate-minus-Ruri differences:

| Metric | Difference | 95% interval |
|---|---:|---:|
| Accuracy | +20.26 pp | [+17.57, +22.92] pp |
| NLL | −0.4288 | [−0.4738, −0.3827] |
| Brier | −0.2082 | [−0.2306, −0.1852] |

The interval is descriptive and does not account for checkpoint selection on
the same validation. Reloading the selected BERT readout reproduced all 2,004
argmaxes; maximum probability delta was `0.0` in the paired comparison verifier.

## Model, compute, and artifacts

- Encoder: [`tohoku-nlp/bert-base-japanese-v3`](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3), pinned revision `65243d6e5629b969c77309f217bd7b1a79d43c7e`; declared Apache-2.0.
- Pretraining-only base: 12 layers, hidden size 768, 111,207,168 frozen parameters. The loaded encoder excludes unused MLM/NSP task heads.
- Weight file: 447,406,217 bytes; SHA-256 `e172862e0674054d65e0ba40d67df2a4687982f589db44aa27091c386e5450a4`.
- Environment: Windows, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, `fugashi` 1.5.2, `unidic-lite` 1.0.8, four CPU threads. RTX 3080 was left unused because it was occupied; no cloud compute was used.
- JNLI: 702.94 s feature extraction and 4.72 s readout fitting; observed process RSS peaked at about 4.9 GB.
- JCommonsenseQA: 905.19 s feature extraction and 3.17 s readout fitting; observed process RSS peaked at about 6.1 GB.
- Readout, feature caches, validation predictions, run metadata, split files, and model weights remain outside Git under `C:\CodexArtifacts\pretrained-reuse-r2-local\`.
- Executed scripts were snapshotted beside the run artifacts. Source commit before this probe was `4476fe1abceaa993c82200301fd2658393db728e`.

The JNLI split generator uses pinned source hashes and refuses inputs other than
the exact publisher train/dev filenames. The test and scoring code rejects
split-hash mismatch and, in comparison, rejects misaligned sample IDs, targets,
or groups. No fitting was done on the official JNLI test (which was not
obtained) or any sealed audit.

## Decision and next step

Keep the Japanese BERT-base family in the text candidate pool. The result
supports further work on text Decision quality; it does not justify adding
111M parameters to the current multimodal assembly. First check seed stability
on the frozen JCommonsenseQA protocol, then evaluate a separately sourced,
rights-cleared Japanese decision task without repeatedly tuning this exposed
JGLUE validation. In parallel, retain the four-modality gate: this BERT probe
has no image, audio, video, or sensor inputs, and its accuracy is below the
project's 95–98% aspiration.

## Reproduction

Prepare JNLI split:

```powershell
python scripts/prepare_jnli_grouped_split.py `
  --publisher-train <pinned-JGLUE>/datasets/jnli-v1.1/train-v1.1.json `
  --publisher-dev <pinned-JGLUE>/datasets/jnli-v1.1/valid-v1.1.json `
  --output-dir <new-empty-artifact-dir> --seed 17
```

Training and paired-comparison commands, exact arguments, hashes, selected
epochs, per-epoch curves, predictions, and checkpoint receipts are recorded in
the external run reports and the [manifest](../../manifests/pretrained_reuse/japanese-bert-readout-probes-20261123.json).

## Sources

- [Official JGLUE repository and task documentation](https://github.com/yahoojapan/JGLUE)
- [Japanese BERT v3 model card](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)
