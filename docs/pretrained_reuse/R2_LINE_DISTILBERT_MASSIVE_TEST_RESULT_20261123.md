# LINE DistilBERT Japanese MASSIVE test result — 2026-11-23

## Result and interpretation

The fixed, one-time publisher-test evaluation completed on 2,815 Japanese
MASSIVE test examples after a predeclared train-overlap filter. The candidate
is a frozen first-four-layer LINE DistilBERT encoder plus a small fixed-task
60-intent readout. It reached **79.01% sample Accuracy** and **70.64% macro
Accuracy across the 59 represented intents**. This is useful evidence that
the truncated BERT-family encoder retains signal for this narrow Japanese
intent-classification task, but it is well below the project's 95–98% target
and does not test arbitrary supplied options, general text reasoning, or
multimodal fusion.

The result does not justify replacing the general text Decision path or
continuing a broad BERT sweep. It is a single task and candidate, with no
validation-based checkpoint selection or validation curve. The training CE
fell throughout eight epochs, but without a validation curve this cannot
establish absence of overfitting. The test is now exposed and must not be used
for additional selection, tuning, or repeat evaluation. The separate project
sealed audit was not opened.

| Metric | Result |
|---|---:|
| Scored rows | 2,815 / 2,974 (159 exact normalized-utterance overlaps with train removed) |
| Unique normalized-utterance groups | 2,777 |
| Sample Accuracy | 0.7901 |
| Macro Accuracy over represented intents | 0.7064 (59 of 60 intents) |
| NLL | 0.7250 |
| Multiclass Brier | 0.2970 |
| ECE, 15 equal-width confidence bins | 0.0203 |
| Mean confidence | 0.7743 |
| Group-bootstrap 95% Accuracy interval | [0.7750, 0.8066] |

The interval resampled 2,777 normalized-utterance groups with replacement,
2,000 replicates, seed 17, using percentile endpoints. It is a descriptive
uncertainty diagnostic on this now-exposed test, not a new blind evaluation.

## Frozen run and provenance

- Protocol: [one-time test protocol](R2_LINE_DISTILBERT_MASSIVE_TEST_PROTOCOL_20261123.md).
- Source commit for the completed run: `5455ba76c4d18b8a1e171ab3805a5b4da015abbf`.
- Dataset: `AmazonScience/massive`, pinned revision
  `5092332a2835c4d29c617ef6d118123a1abecd19`, declared CC BY 4.0.
- Test file: `ja-JP/test-00000-of-00001.parquet`, 220,619 bytes,
  SHA-256 `985d95a0ec20724bc8f07e17e375f73ced6572f9670a3a525a19065736cde012`.
- Frozen train JSONL: 11,228 rows, SHA-256
  `ab55ddd41e4cf0f07f65bddc84432bea40ceb36caaa94c6edf15499ddf6a1270`.
- Split manifest SHA-256
  `d9fdcc0235b3caecac462574e6111a6c1bf59102eab3ce39361c7fb42af9ed1d`.
- Test preflight: zero source-ID overlap; 159 normalized-content overlaps
  were excluded before fitting; ordered IDs of the 2,815 scored rows hash to
  `c612850a105461c6e0393da0803014be41edb89ae668a87f8546e60f4e895a12`.
- Encoder: `line-corporation/line-distilbert-base-japanese`, revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`; source weight file SHA-256
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`.
- First four of six pretrained layers; 53,912,064 frozen encoder parameters.
  The fixed 768→128 tanh/dropout→60 readout has 106,172 trainable parameters,
  for 54,018,236 parameters in this isolated task model.
- Seed 17, float32 frozen features, batch 64, exactly eight epochs, AdamW
  learning rate 0.002 and weight decay 0.0001. Epoch 8 was fixed in advance;
  no validation selection, calibration, or test-guided changes were used.
- CPU-only environment: Python 3.11.9, PyTorch 2.6.0+cu124,
  Transformers 5.6.2, four CPU threads, CUDA unavailable to the process.
  Encoder feature extraction took 66.36 s, readout fitting 4.01 s, test
  feature extraction 17.15 s, and readout scoring 0.003 s. Peak process
  working set was 2,230,091,776 bytes.
- Final run artifacts total 39,636,042 bytes (37.8 MiB), including the
  34.6 MB frozen training-feature cache, readout checkpoint, history, and
  predictions. This is not a quantized product-package measurement.

The authoritative report, all per-intent metrics, prediction probabilities,
training history, and hashes remain outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\massive-ja-line-distilbert4-official-test-20261123-attempt-02\`.
Its `run-report.json` SHA-256 is
`3f4d87a4f15d793314262f426ed0ff423ebbe1729b61fbadecf32769a10ad4c`.
The report records script SHA-256
`37d5639bd6bd820a6f25a9a130196dda844768aa0e7c04ed64777efc8922c747`.
The initial attempt stopped before reading corpus data because Windows
PyTorch still reported CUDA available with an empty visibility value; it
recorded no labels or scores. A code-only correction set
`CUDA_VISIBLE_DEVICES=-1` and enforced CPU execution. The completed run used a
new output directory and did not alter model weights, frozen corpus, other
experiments, or the live Orbitune process.

## Per-intent results

Accuracy is reported for each represented intent; `—` means the publisher
test contained no scored row for that intent. The full probability vector for
every example is preserved in the external prediction artifact.

| Intent | N | Accuracy |
|---|---:|---:|
| datetime_query | 78 | 94.9% |
| iot_hue_lightchange | 34 | 82.4% |
| transport_ticket | 34 | 91.2% |
| takeaway_query | 35 | 71.4% |
| qa_stock | 25 | 92.0% |
| general_greet | 1 | 0.0% |
| recommendation_events | 43 | 79.1% |
| music_dislikeness | 4 | 50.0% |
| iot_wemo_off | 14 | 57.1% |
| cooking_recipe | 72 | 90.3% |
| qa_currency | 37 | 75.7% |
| transport_traffic | 15 | 86.7% |
| general_quirky | 166 | 60.2% |
| weather_query | 144 | 94.4% |
| audio_volume_up | 8 | 50.0% |
| email_addcontact | 11 | 45.5% |
| takeaway_order | 22 | 77.3% |
| email_querycontact | 26 | 73.1% |
| iot_hue_lightup | 23 | 47.8% |
| recommendation_locations | 31 | 80.6% |
| play_audiobook | 41 | 63.4% |
| lists_createoradd | 35 | 80.0% |
| news_query | 114 | 74.6% |
| alarm_query | 27 | 92.6% |
| iot_wemo_on | 10 | 40.0% |
| general_joke | 18 | 88.9% |
| qa_definition | 57 | 82.5% |
| social_query | 24 | 87.5% |
| music_settings | 5 | 20.0% |
| audio_volume_other | 6 | 0.0% |
| calendar_remove | 61 | 90.2% |
| iot_hue_lightdim | 16 | 43.8% |
| calendar_query | 120 | 56.7% |
| email_sendemail | 110 | 91.8% |
| iot_cleaning | 23 | 69.6% |
| audio_volume_down | 9 | 55.6% |
| play_radio | 69 | 87.0% |
| cooking_query | 0 | — |
| datetime_convert | 14 | 57.1% |
| qa_maths | 23 | 91.3% |
| iot_hue_lightoff | 32 | 90.6% |
| iot_hue_lighton | 2 | 0.0% |
| transport_query | 51 | 68.6% |
| music_likeness | 36 | 75.0% |
| email_query | 116 | 84.5% |
| play_music | 171 | 81.9% |
| audio_volume_mute | 29 | 79.3% |
| social_post | 81 | 82.7% |
| alarm_set | 38 | 65.8% |
| qa_factoid | 138 | 84.1% |
| calendar_set | 205 | 91.2% |
| play_game | 32 | 87.5% |
| alarm_remove | 18 | 61.1% |
| lists_remove | 48 | 87.5% |
| transport_taxi | 22 | 90.9% |
| recommendation_movies | 19 | 73.7% |
| iot_coffee | 30 | 83.3% |
| music_query | 33 | 78.8% |
| play_podcasts | 59 | 47.5% |
| lists_query | 50 | 82.0% |
