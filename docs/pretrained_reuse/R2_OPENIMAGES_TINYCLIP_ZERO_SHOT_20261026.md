# Open Images V6 TinyCLIP zero-shot option evaluation — 2026-10-26

## Scope

This is one fixed-prompt, no-training Path A transfer diagnostic on the already
exposed Open Images V6 development validation. It asks whether a compact
pretrained image/text representation can rank the supplied object options from
the actual image. It is not a blind result, a product acceptance result, or a
four-modality quality claim. The official test split and sealed data were not
loaded.

The existing page-matched validation manifest had 171 examples. Three media
files were unavailable across the two preserved download attempts, leaving 168
examples with saved image bytes, matching Flickr CC BY 2.0 page declarations,
and matching attribution profiles. The source audit explicitly did not
independently verify image ownership. Those three unavailable records were
excluded by the pre-existing media-success gate; labels, options, split, and
prompt were not changed. The original download attempts and their files were
left untouched.

## Frozen model and decision rule

- Checkpoint: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, revision
  `07a4b0bc751cb64fecd2b661c048c1dd98d69444`.
- Model card declares MIT. The checkpoint was already present locally from the
  R1 metadata/reuse work; no weights were downloaded or changed for this run.
- Runtime count: **83,101,185 unique parameters**. The `model.safetensors`
  file alone is **332,443,148 bytes**, above the provisional 200 MB package
  target before tokenizer and runtime files are included.
- Weights SHA-256:
  `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`.
- Fixed prompt: `a photo containing {lowercase option}.` One embedding is
  computed per unique option; each image is ranked only among its four supplied
  choices using the checkpoint's image-text similarity logits and softmax.
- No training, prompt search, calibration, checkpoint selection, or GPU use.

## Results

| Input condition | n | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Correct image and options | 168 | 64.88% | 0.9902 | 0.4899 | 0.0942 | 0.7409 |
| Neutral gray image, same options | 168 | 49.40% | 1.0837 | 0.6119 | 0.1531 | 0.6001 |
| Deranged validation images, same options | 168 | 34.52% | 1.6581 | 0.8377 | 0.2692 | 0.6144 |

The deterministic image derangement used seed `20261026` and has zero fixed
points. Correct-image top-1 predictions agreed with the neutral-image control
on 66/168 rows and with the deranged-image control on 52/168 rows. Correctness
changed on 76 and 83 rows, respectively. Thus this TinyCLIP run is sensitive to
the supplied image and its pairing with the choices. Image sensitivity alone
does not make its 64.88% accuracy high or generalizable.

Per-target results (each row is a small subset of the same exposed validation):

| Target | n | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| Toy | 16 | 100.0% | 0.120 | 0.050 | 0.100 |
| Door | 14 | 64.3% | 0.660 | 0.390 | 0.160 |
| Food | 15 | 93.3% | 0.210 | 0.100 | 0.100 |
| Plant | 16 | 50.0% | 1.550 | 0.710 | 0.320 |
| Sculpture | 15 | 93.3% | 0.180 | 0.080 | 0.120 |
| Vehicle | 18 | 61.1% | 0.780 | 0.420 | 0.190 |
| Clothing | 16 | 37.5% | 1.880 | 0.890 | 0.340 |
| Furniture | 17 | 52.9% | 1.250 | 0.700 | 0.280 |
| Building | 13 | 15.4% | 2.390 | 1.180 | 0.520 |
| Window | 15 | 66.7% | 1.230 | 0.560 | 0.230 |
| Animal | 13 | 76.9% | 0.730 | 0.360 | 0.240 |

## Comparison and interpretation

On the same 168 exposed validation IDs, the earlier fitted VideoPrism +
MiniLM readout reported 79.17% accuracy, NLL 0.5609, Brier 0.3014, and ECE
0.1454; its zero-image and deranged-image controls left all top-1 predictions
unchanged. A separate fitted MiniLM-only readout reported 75.60% accuracy.
Those are not capacity- or training-matched comparisons. TinyCLIP is lower in
accuracy than both fitted readouts, while its own controls show that correct
image identity affects its decisions. This is evidence for image-conditioned
option ranking on this narrow task, not evidence that TinyCLIP is the best
Path A candidate.

The small per-target counts (13–18), prior exposure of this validation, and
remaining uncertainty in source-image ownership block broad quality claims.
The model has only 168 evaluated examples; high rates for a few categories are
not stable class-quality estimates. The overall result is well below the
project's 95–98% aspiration. The 332.4 MB weights file also exceeds the
provisional package target before compression.

## Reproduction and saved outputs

The effective settings are in
[`path_a_openimages_tinyclip_zero_shot.yaml`](../../configs/pretrained_reuse/path_a_openimages_tinyclip_zero_shot.yaml).
The evaluator is
[`evaluate_openimages_tinyclip_zero_shot.py`](../../scripts/evaluate_openimages_tinyclip_zero_shot.py)
(SHA-256 `e257149d0f6c5bef9b8d780354dca74683d84cd11cfc8d5a4e5b265b8e6d99ce`).
It strictly checks the model checkpoint hash, source-media hashes, split and
row ordering, duplicate identities, option count, and the per-image page
declaration gate.

The invocation was:

```powershell
python scripts/evaluate_openimages_tinyclip_zero_shot.py `
  --features-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-videoprism-features-20261026-attempt-01 `
  --media-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-media-cache-20261026-attempt-01 `
  --media-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-media-cache-20261026-attempt-02 `
  --model-dir C:\CodexArtifacts\pretrained-reuse-r1\tinyclip-39m19m-audit-01\model `
  --tasks-manifest C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-support-audit-20261026\page-matched-validation-tasks.json `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\openimages-tinyclip-zero-shot-20261026-attempt-02 `
  --batch-size 8 --image-controls
```

Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2; inference ran
on CPU with four threads. Aligned forward time was 10.23 seconds and the two
image controls took another 19.99 seconds. No GPU memory or paid compute was
used. The result JSON SHA-256 is
`0626cb6b0ca1b9fdafbc0cb6b907e6e0e430e6c3f2023d49c7ade48b89ccf6d7`;
prediction JSONL SHA-256 is
`7ed8e6877b78541effb45d2ee5cdb4ad6184609eee0e7e654d1eddd27f39e711`;
control prediction JSONL SHA-256 is
`e96b6c45bb2d17d6e891237237e2a414980df34c16fc1bf9b7bfb4d62716c075`.
These outputs remain outside Git. The fixed task manifest SHA-256 is
`6fbd56f0caebc77b1067dfffaadb444aaf51b4b01eabff3855c11a245f86f964`.
The aligned VideoPrism validation-row JSONL SHA-256 is
`c2396b3974de91afea66424b201388fb6d47013098a211d31b0191bd0a9adfb7`.
The page-matched media validation manifests for download attempts 01 and 02
have SHA-256 values `81b225f38d311ffe94003b2419100b33f88ad484d7f0f4ab1f3f35877e048560`
and `b854195055b47201fe6b7c890abda032ae0360636563b38069637617e2d2dd3f`,
respectively.

Focused tests: 6 passed; Ruff and `git diff --check` passed. No training, test
split evaluation, sealed audit, model promotion, or data/model artifact was
added to Git.
