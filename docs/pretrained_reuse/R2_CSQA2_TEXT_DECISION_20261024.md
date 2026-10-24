# CommonsenseQA 2.0 text-only Decision probe (2026-10-24)

## Purpose and scope

This is a narrow real-text Path A probe for the repository's typed-option
Decision interface. It does not qualify general text reasoning or the multimodal
product. We used the official CC BY 4.0 training and development files from
AllenAI's [CommonsenseQA 2.0 repository](https://github.com/allenai/csqa2). The
published test file was not downloaded or evaluated. The official development
split selected the checkpoint, so these are development results, not blind
evaluation results.

The CSQA2 task is adversarial yes/no commonsense verification. It is useful for
testing whether the existing frozen text representation plus small option
scorer transfers to a real-language binary Decision task. It is not broad
coverage of arbitrary text decisions. The released rows do not expose author or
participant group IDs, so group-disjoint uncertainty estimates are unavailable.

## Frozen protocol

- Source commit: `3f57a4a6a1d51fe6a9b462b59be67c08d41fdc39`.
- Train/dev records: 9,264 / 2,541; all source rows retained. Train classes are
  balanced 4,632/4,632; dev has 1,316 `no`, 1,225 `yes`.
- Stable options: `["no", "yes"]`; question text is the context and source answer
  is the target. Repeated rows are not discarded.
- Train/dev ID and normalized-question overlaps: zero. Unique normalized
  questions: 9,259 train and 2,540 dev (five train rows and one dev row duplicate
  a normalized question within their own split).
- Encoder: frozen `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; 17,389,824 frozen parameters;
  weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Readout: existing Path A MLP over `[context, option, abs(context-option),
  context*option]`, 128 GELU hidden units and one score per option; 196,865
  trainable parameters. Both candidates are scored and softmaxed to JSON option
  probabilities.
- Seed 17, 8 epochs, batch size 64 questions, AdamW learning rate 0.001 and
  weight decay 0.0001. Select minimum dev NLL. No prompt search, calibration,
  seed search, data filtering, or test-set use.
- CPU only. Frozen feature extraction: 30.60 s; head fitting: 5.85 s. No GPU
  training or cloud compute was used.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, Windows 10;
  CPU reported as `Intel64 Family 6 Model 154 Stepping 3`. CUDA was available
  in the environment but unused.

## Results

The selected checkpoint is epoch 1. Metrics are computed over the 2,541 official
development records using 15-bin ECE.

| Epoch | Train CE | Dev Accuracy | Dev NLL | Dev Brier | Dev ECE |
|---:|---:|---:|---:|---:|---:|
| 1 (selected) | 0.6936 | 0.5179 | 0.6925 | 0.4994 | 0.0116 |
| 2 | 0.6909 | 0.5207 | 0.7109 | 0.5171 | 0.0959 |
| 3 | 0.6874 | 0.5081 | 0.7012 | 0.5079 | 0.0516 |
| 4 | 0.6810 | 0.5077 | 0.6990 | 0.5056 | 0.0565 |
| 5 | 0.6774 | 0.5163 | 0.6977 | 0.5044 | 0.0465 |
| 6 | 0.6718 | 0.5195 | 0.7020 | 0.5080 | 0.0650 |
| 7 | 0.6699 | 0.5104 | 0.7068 | 0.5129 | 0.0730 |
| 8 | 0.6648 | 0.5207 | 0.7143 | 0.5195 | 0.0869 |

The selected model's accuracy equals the dev majority-class baseline (51.79%).
It predicted `no` for 2,377/2,541 records (93.5%). Training CE fell while dev
NLL, Brier, and ECE mostly degraded after epoch 1. This is evidence of poor
transfer and a train/dev divergence for this fixed setup, not evidence that the
encoder or model family has reached a general ceiling. Low epoch-1 ECE reflects
near-uniform probabilities and does not indicate useful discrimination.

Candidate-order invariance passed: maximum option-identity logit difference
after swapping options was `2.98e-8`. Output follows the existing typed JSON
probability contract.

## Reproducibility artifacts

All run outputs are outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/csqa2-text-decision-v0-attempt-02/`.
The feature cache is 148,378,818 bytes and remains outside the repository.

| File | SHA-256 |
|---|---|
| train gzip source | `cdaccded500ea4093ee88ee28d9f61f2b210ee09c6b996d1242d66aeac0b3b32` |
| dev gzip source | `2804671317521bb79784037482b7e84e891c707f6ac369607209059b3df3b884` |
| best readout | `a71ba569925562dfb7b8e9f72f063c7c7e0657aa0e0341bfbda5839df38a7ced` |
| training history | `b1b3b5c10b86cd4b7028db3d4fbb73867de9034dfcc8d2f6099b9b97d7206b88` |
| dev predictions | `6d5e3eaf6f11d7493bcdf755b5d21ea4fe4705dd2556497ba7785d4cb2195a26` |
| run report | `f3f3a5ddb1fd4f7e6b38d2937e7e35d4cf023ee2dc61a2a018a919a18044e84a` |

The run report also records manifest, model, feature-cache, runner and prediction
hashes, full epoch metrics, exact settings and output counts.

## Interpretation and next step

This small head did not extract useful CSQA2 discrimination from the fixed
MiniLM feature contract. The model is strongly biased toward `no` on dev despite
balanced train labels. Potential causes include limitations in the pooled
sentence embedding, the candidate-text representation, the readout objective,
CSQA2's adversarial data distribution, or ordinary generalization failure; this
experiment does not isolate them. Do not increase epochs or tune on this dev
split. Preserve the result as a failure reference. A next text experiment should
pre-register a distinct representation/readout intervention and an independent
evaluation source before fitting.

No product model was promoted. No existing Teacher, video, audio, image, audit,
or QAT artifact was modified.
