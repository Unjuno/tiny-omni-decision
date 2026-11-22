# Japanese MASSIVE intent readout (2026-11-23)

## Result

The frozen Japanese BERT-base text path reached **81.65% accuracy** on the
publisher's Japanese MASSIVE development split after removing normalized
utterance duplicates from the training split. This is a useful independent-
source text-transfer result after the JGLUE probes, but remains below the
project's 95–98% aspiration. It provides no evidence for image, audio, video,
sensor, or multimodal fusion quality.

The result is checkpoint-selection-affected: validation NLL selected epoch 8
from eight epochs, and the metrics below are reported on that same validation
set. The publisher test split was not downloaded or opened. No project sealed
audit was accessed.

| Epoch | Train CE | Accuracy | Macro Accuracy (59 present intents) | NLL | Brier | ECE-15 | Mean confidence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.8112 | 75.60% | 63.62% | 0.9910 | 0.3855 | 0.1504 | 0.6065 |
| 2 | 0.8046 | 80.08% | 71.10% | 0.7678 | 0.3021 | 0.0575 | 0.7439 |
| 3 | 0.6457 | 79.69% | 70.98% | 0.7301 | 0.2914 | 0.0259 | 0.7743 |
| 4 | 0.5340 | 80.87% | 72.76% | 0.7039 | 0.2806 | 0.0273 | 0.7898 |
| 5 | 0.4674 | 80.87% | 70.68% | 0.7157 | 0.2820 | 0.0249 | 0.8202 |
| 6 | 0.4108 | 81.46% | 75.37% | 0.7018 | 0.2756 | 0.0181 | 0.8224 |
| 7 | 0.3661 | 81.55% | 76.59% | 0.6896 | 0.2749 | 0.0349 | 0.8343 |
| 8 (selected) | 0.3246 | 81.65% | 75.24% | 0.6872 | 0.2730 | 0.0360 | 0.8385 |

The selected checkpoint's accuracy is 81.65%, macro accuracy over the 59
validation-present intents is 75.24%, NLL 0.6872, Brier 0.2730, ECE-15 0.0360,
and mean confidence 0.8385 (2,033 examples). `audio_volume_other` has no
validation examples. Some other intents have only two or five examples, so
their per-intent rates are noisy; full counts, metrics, and per-example
probabilities are in the external run report and prediction file. Macro
accuracy fell from 76.59% at epoch 7 while weighted accuracy/NLL improved at
epoch 8, and ECE increased. This is a class-balance/calibration tradeoff, not
uniform improvement across intents.

Train CE continued to fall throughout the run while validation NLL also fell
from 0.9910 to 0.6872. There is no validation-NLL rise in this eight-epoch
window, but the macro-accuracy and ECE changes at the last epoch caution
against interpreting aggregate accuracy as uniform quality. The observed
validation split has already been used for model selection and must not be
treated as blind evidence or repeatedly tuned against.

## Frozen protocol and integrity

- Source: AmazonScience MASSIVE, pinned revision
  `5092332a2835c4d29c617ef6d118123a1abecd19`, Japanese `ja-JP`, declared
  CC BY 4.0. The source card identifies the task as intent classification and
  the corpus as localized SLURP utterances.
- Only publisher train and validation Parquet files were downloaded. Their
  SHA-256 hashes are `f1404ede08f40ffe4fd64f72adb312304ab7cb5afa1d0bdf4fc61e518bc49de8`
  and `36283fc28102e6c232ce6928917fa31b898c5d678f18f90f411a086c008c1cc9`.
- The pinned train/dev files contained 286 training rows whose NFKC,
  casefolded, whitespace-collapsed utterance exactly matched a validation
  utterance. Those training rows were removed. Final counts: 11,228 train and
  2,033 validation, with 60/60 and 59/60 intent coverage respectively.
- Cross-split source-ID overlap is zero. Cross-split normalized utterance
  overlap is zero. The publisher test and project sealed audit remain
  unopened. Annotated utterances, slots, worker IDs, and judgments were not
  model inputs.
- Within the final train split, 11,228 rows contain 10,891 unique normalized
  utterances (337 repeats after first occurrence); 14 repeated-text groups
  (34 rows) have conflicting intent labels. Validation contains 2,033 rows and
  2,015 unique normalized utterances, with 18 repeats and no conflicting-label
  duplicate groups. These remain in the source-defined split and are disclosed
  as data ambiguity; they were not edited or selectively removed.
- The input is a single utterance; the model predicts one of the fixed 60
  intent labels and stores a probability vector. This is not free-form
  candidate-option scoring, a generated JSON runtime, or a multimodal test.

## Model and execution

- Frozen `tohoku-nlp/bert-base-japanese-v3`, revision
  `65243d6e5629b969c77309f217bd7b1a79d43c7e`, Apache-2.0; 111,207,168 encoder
  parameters, 447,406,217-byte FP32 weight file.
- Readout: 768 → 128 tanh → dropout 0.1 → 60 logits; 106,172 trainable
  parameters and 111,313,340 total pre-quantization parameters.
- Seed 17, 8 epochs, batch 64, max token length 128, AdamW learning rate
  0.002 and weight decay 0.0001. Best checkpoint selected by validation NLL.
- CPU-only on Windows, Python 3.11.9, PyTorch 2.6.0+cu124,
  Transformers 5.6.2, fugashi 1.5.2, unidic-lite 1.0.8. Feature extraction
  took 200.33 seconds; readout fitting took 3.32 seconds. No GPU or cloud
  compute was used for this run.
- The selected readout was saved and reloaded with `weights_only=True`; the
  run receipt records successful reload verification. The base model's unused
  MLM/NSP task-head weights were reported as unexpected when loading the BERT
  encoder, as expected for this architecture conversion.
- Source commit before the experiment: `d42653ceb21acbc96171608145664eef80e767a8`.

Exact artifacts are external to Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\`:

| Artifact | SHA-256 |
|---|---|
| train JSONL | `ab55ddd41e4cf0f07f65bddc84432bea40ceb36caaa94c6edf15499ddf6a1270` |
| validation JSONL | `aa42d55cb8afcf7170483322e6a129acd4657a9acf3ea3530fea265a36679707` |
| split manifest | `d9fdcc0235b3caecac462574e6111a6c1bf59102eab3ce39361c7fb42af9ed1d` |
| frozen features | `3177c8d3c29aa2b2519334de53045eb9ba049fc696db01b290f95a21a811f0fc` |
| selected readout | `49d22f13ac6a4b4f3f31798d70d986d815950eb4e4460cb30cc34ab7e012254b` |
| validation predictions | `8c675c3afe1893bec47b53214678935964d4e06931e4a3698a3abc2772319040` |
| run report | `6f5842e2d12b87585dc937305a55824191ec90fbf29041a8eb85a42c690b0312` |

The machine-readable summary is
[`massive-ja-bert-readout-20261123.json`](../../manifests/pretrained_reuse/massive-ja-bert-readout-20261123.json).

## Decision

Keep Japanese BERT-base as a text-path candidate: it transfers to a distinct
Japanese assistant-intent source and reaches a materially useful but
non-final 81.65% development accuracy. Do not describe it as a 95–98% model,
or as a solution to the Omni goal. The low-support intents, the macro/weighted
metric tradeoff, and checkpoint-selection exposure require stronger
independent text evaluation. The product still needs credible image, audio,
video, and cross-modal decision evidence, JSON runtime validation, and
quantization/package/latency measurements. No quantization, compression, or
promotion was performed here.

## Reproduction

Using the pinned local dataset/model and tokenizer dependency directories, the
run used seed 17, eight epochs, batch size 64, encode batch size 64, and a
fresh output directory. The official test was not part of the command:

```powershell
$env:PYTHONPATH = '<tokenizer-dependency-directory>'
$env:CUDA_VISIBLE_DEVICES = ''
python scripts/prepare_massive_ja_intents.py `
  --publisher-train '<massive>/ja-JP/train-00000-of-00001.parquet' `
  --publisher-dev '<massive>/ja-JP/validation-00000-of-00001.parquet' `
  --output-dir '<new-external-split-directory>'
python scripts/train_massive_ja_bert_readout.py `
  --train '<split>/train.jsonl' --validation '<split>/validation.jsonl' `
  --split-manifest '<split>/split-manifest.json' --model '<pinned-local-model>' `
  --output-dir '<new-external-run-directory>' --seed 17 --epochs 8 `
  --batch-size 64 --encode-batch-size 64
```
