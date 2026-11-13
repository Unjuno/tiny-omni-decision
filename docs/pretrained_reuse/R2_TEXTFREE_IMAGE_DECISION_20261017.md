# Text-free CLEVR-4 image Decision probe — 2026-10-17

## Scope

This one-shot Path A probe measures a small readout's ability to make four
closed-set taxonomy decisions from image features alone. It is a direct
image-only M-0/M-1 implementation milestone. It is not a natural-image
reasoning, video, or product-quality result. No question text, candidate text,
audio feature, text encoder, official validation labels/features, or sealed
audit was loaded during training or scoring.

The internal development split was deterministically carved from the official
CLEVR-4 **training** images. It was not blind: an earlier full-coverage
image+text readout trained on all official training records. No cross-model
comparison is made here. This result must not be treated as independent
evidence or used to assert a general image-accuracy target.

## Inputs and integrity

- Source: `sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73`,
  manifest license `CC-BY-4.0`; the task is synthetic image taxonomy
  classification, not natural-image reasoning.
- Train records: 8,424 image IDs and 8,424 unique image SHA-256 values.
- Labels and media manifest have matching ID order. Cached feature IDs and
  media SHA-256 values exactly match those manifests; feature rows are finite
  float32 `[8424, 768]` vectors.
- Train labels SHA-256:
  `00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e`.
- Train image manifest SHA-256:
  `2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010`.
- Frozen feature cache SHA-256:
  `f36b9cf17efed6a5137eb48ffcaf40ff960015fc07bc194fb532bf5562d88fab`.
- Cache report SHA-256:
  `f8789fed749c8e19741758d52af2468b74598148dc673174ba65fb86181d5467`.
- VideoPrism-B `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`,
  implementation commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`,
  114,365,184 encoder parameters. Each still image uses the existing cached
  8-slot repeated-frame encoding; that does not provide temporal evidence.
- The original 3.8 GB source archive's full SHA-512 was **not** verified by the
  existing extraction receipt. Per-image manifest hashes and feature alignment
  did pass; the archive-level limitation remains open.

The script opened only `train_ids`, `train_media_sha256`, and `train_features`
from the NPZ. It did not index or score its official validation arrays and did
not read `val-labels.jsonl` or `val-image-manifest.jsonl`.

## Fixed recipe

Seed 17; four independent 768→10 linear heads (30,760 trainable parameters);
AdamW, learning rate `1e-3`, weight decay `1e-4`; batch size 256; constant
learning rate; 128 epochs with no performance early stopping. A SHA-256 ordering
of media identities assigned 842 images to internal development and 7,582 to
training, with zero image/hash overlap. All four taxonomies were present in
both splits. Every epoch consumed each training image once, with four targets
scored per image.

The checkpoint was selected by lowest macro NLL across the fixed 128-epoch
internal-development curve. The best and final state files remain separate;
both selected epoch 128. Save/reload reproduced selected development logits
exactly. There were 3,840 optimizer updates, 7,582 unique training images, and
970,496 image exposures over the full run (962,914 repeats after first use).
At inference, the readout normalizes over any supplied set of at least two
known class IDs; the recorded evaluation uses all ten classes for each
taxonomy.

## Results

Metrics below use 842 held-out training images per taxonomy. Macro is the
unweighted mean across color, texture, count, and shape.

| Taxonomy | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Color | 0.9941 | 0.0623 | 0.0178 | 0.0460 |
| Texture | 0.9311 | 0.2938 | 0.1244 | 0.1071 |
| Count | 0.5618 | 1.0855 | 0.5551 | 0.0908 |
| Shape | 0.9988 | 0.0114 | 0.0024 | 0.0105 |
| **Macro** | **0.8714** | **0.3633** | **0.1749** | **0.0636** |

Macro mean confidence was `0.8086`. Selected train macro Accuracy/NLL was
`0.8860 / 0.3466`, versus internal-development `0.8714 / 0.3633`; this is a
small fixed-split gap, not evidence of broad generalization. Count is the clear
weak task at 56.18%; the current result cannot distinguish representation,
feature pooling, label/task difficulty, or optimization causes.

| Epoch | Updates | Train macro Accuracy | Train macro NLL | Dev macro Accuracy | Dev macro NLL |
|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 0.5322 | 2.0435 | 0.5211 | 2.0465 |
| 8 | 240 | 0.7742 | 1.1664 | 0.7660 | 1.1724 |
| 16 | 480 | 0.8176 | 0.8444 | 0.8103 | 0.8506 |
| 32 | 960 | 0.8439 | 0.6050 | 0.8388 | 0.6104 |
| 64 | 1,920 | 0.8646 | 0.4485 | 0.8539 | 0.4583 |
| 96 | 2,880 | 0.8773 | 0.3840 | 0.8658 | 0.3975 |
| 120 | 3,600 | 0.8844 | 0.3547 | 0.8679 | 0.3703 |
| 128 | 3,840 | 0.8860 | 0.3466 | 0.8714 | 0.3633 |

Accuracy and NLL were still improving at the final scheduled epoch. This
128-epoch run did **not** establish a plateau. The subsequent 512-epoch run
below starts from a new seed-17 initialization and preserves this artifact;
it is not an optimizer resume.

## Longer-budget rerun — 512 epochs

The second run retained seed 17, the same model, train images, media-hash split,
batch size, AdamW settings, constant learning rate, and per-epoch evaluation.
Only the planned epoch budget and output ID changed. It started from fresh
random linear heads, not either saved checkpoint. The first 128 epochs match
the 128-epoch run exactly at every saved training and validation metric and
sample-order hash. Between runs, only the inference-time candidate validation
was tightened; the model forward, loss, optimizer, and sampler behavior were
unchanged.

The selected checkpoint is epoch 502 (15,060 optimizer updates); final epoch
512 is retained separately. The fixed selector is still minimum macro
development NLL. This is a longer-budget comparison on the same reused
internal development IDs, not an independent test.

| Taxonomy | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Color | 0.9964 | 0.0110 | 0.0058 | 0.0050 |
| Texture | 0.9537 | 0.1478 | 0.0687 | 0.0336 |
| Count | 0.5819 | 0.9447 | 0.5062 | 0.0379 |
| Shape | 1.0000 | 0.0010 | 0.0002 | 0.0010 |
| **Macro** | **0.8830** | **0.2761** | **0.1452** | **0.0194** |

At final epoch 512, macro Accuracy/NLL/Brier/ECE-15 was
`0.8821 / 0.2764 / 0.1454 / 0.0172`. Best epoch 502's train macro
Accuracy/NLL was `0.9175 / 0.2314`, a 3.45-point Accuracy gap and 0.0447 NLL
gap to its internal development split. Training NLL continued to fall through
512 while development NLL was marginally worse than its epoch-502 minimum;
this is a mild late overfitting/plateau signal, not a clean ability ceiling.

| Epoch | Updates | Train macro Accuracy | Train macro NLL | Dev macro Accuracy | Dev macro NLL |
|---:|---:|---:|---:|---:|---:|
| 128 | 3,840 | 0.8860 | 0.3466 | 0.8714 | 0.3633 |
| 256 | 7,680 | 0.9029 | 0.2783 | 0.8726 | 0.3080 |
| 384 | 11,520 | 0.9128 | 0.2479 | 0.8765 | 0.2868 |
| 448 | 13,440 | 0.9147 | 0.2384 | 0.8809 | 0.2803 |
| 480 | 14,400 | 0.9170 | 0.2341 | 0.8806 | 0.2789 |
| 496 | 14,880 | 0.9169 | 0.2321 | 0.8821 | 0.2770 |
| **502 selected** | **15,060** | **0.9175** | **0.2314** | **0.8830** | **0.2761** |
| 512 final | 15,360 | 0.9172 | 0.2304 | 0.8821 | 0.2764 |

Compared with the same run's epoch 128, selected epoch 502 increases macro
Accuracy by 1.16 percentage points and reduces macro NLL by 0.0872. The
weakest count taxonomy improves from 0.5618 to 0.5819 Accuracy, but remains
far below the other three attributes. This supports a training-budget effect
for this narrow development task; it does not isolate the cause of the count
error or imply the 95–98% product target.

Run directory:
`C:\CodexArtifacts\pretrained-reuse-r2\textfree-image-decision-512-01\`.

- Effective config SHA-256:
  `c4118b5026a52bd570fa9f1e7d94c8d47e313a415ebeef3412be8a0b2bc7e0fc`.
- Best readout SHA-256:
  `e62711ca1b8fd68df5037cd135266ed7f556e0ad006359305f10e6665789a47d`.
- Final readout SHA-256:
  `eee0b8ee8d3b2c66337aedc85c36049936807999c4fc2ca1001402f459753911`.
- Training history SHA-256:
  `5d9aa6388b1808646791d36416cf6dc5be129e7bcd57f164b93e853ee3e944ec`.
- Selected predictions SHA-256:
  `b42fa41f3b63fcbac62add2a4e71bc606cbe3276462c2474b8fc61fd3e5f756c`.
- Report SHA-256:
  `f2b357b00bf2ce11cc3c1b6f75b64aa8ea1a9c132908601c1fafc2e98fe80b39`.
- Source receipt SHA-256:
  `66a38685ad4858d3d6bfd3bdded0d2cad736b560f3afaab0d81544d8241c2171`.
- Run wall `83.05 s`; optimizer portion `41.30 s`; epoch/final evaluation
  `35.20 s`; peak PyTorch allocated VRAM `113,812,992` bytes (`108.5 MiB`).
- Full run exposures: 7,582 unique train images, 3,881,984 image exposures
  and 15,527,936 taxonomy-task exposures. The selected checkpoint consumed
  3,806,164 image exposures from the same 7,582 unique train images.

The 512-run source commit is still the base
`facc8665438b425872ffbd1b5ad8b3b72cd28084`; exact script/readout hashes are
bound in the source receipt because these new implementation files were
uncommitted at execution. No raw media was reopened, no official validation or
sealed audit was accessed, and no prior artifact was overwritten.

## Runtime and artifact

- Device: RTX 3080 Laptop GPU (16 GiB); CUDA 12.4 runtime, PyTorch
  `2.6.0+cu124`, Python `3.11.9`, NumPy `2.4.6`, Windows 10 build 26300.
- The frozen VideoPrism encoder was not loaded in this fit; only cached
  features and the 30,760-parameter heads were on the training device.
- Run wall time `42.24 s`; optimizer portion excluding epoch evaluation
  `15.31 s`; epoch and final evaluation `21.67 s`.
- Peak PyTorch allocated VRAM `113,677,824` bytes (`108.4 MiB`) for this
  process. The machine had an unrelated GPU training process during the run;
  `nvidia-smi` showed about 15.8–16.1 GiB board memory in use. The measured
  108.4 MiB is process allocation, not total board memory or end-to-end model
  inference cost. The other process was not manually stopped.
- Pre-quantization image-path parameter arithmetic is
  `114,365,184 + 30,760 = 114,395,944`, excluding unrelated modalities,
  fusion, tokenizer or runtime support. It fits the 200M arithmetic ceiling
  for this isolated path, not for the complete Omni product.
- External run directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\textfree-image-decision-01\`.
- Effective config SHA-256:
  `fe184ca47d5ccede0416ac9f4b5a6441ad2134c11a9f11b5473aea6e75b2840e`.
- Best readout SHA-256:
  `7b4e1e0638716262e2d52b7b94a3ad7c3ce2c8f32a1e7c46aeaabeb2617253d0`.
- Final readout SHA-256:
  `6c362c44f5d4225e58dd4421588a4178815482b7b81f4b035604cc5001820cf1`.
- Training history SHA-256:
  `e5de51924749e01e500f2a4cc48b54ca90b744306978cc7e8b4fda996595d4d0`.
- Selected predictions SHA-256:
  `41f1773eea733287290b781a521d96a43a456b2a4a8d20932ddfe0bde9597b04`.
- The executed source was based on repository commit
  `facc8665438b425872ffbd1b5ad8b3b72cd28084` plus the new uncommitted files
  below. Exact execution-time source hashes are recorded in the external
  `run-source-receipt.json` beside the run report; its SHA-256 is
  `21ca479a86ef84413606e5ab5507af606b31851dd9b8c5d63c8bef1198285432`.

Implementation and tests:

- `src/tiny_omni_decision/textfree_image.py`
- `scripts/train_textfree_image_probe.py`
- `tests/test_textfree_image.py`

## Interpretation

This confirms that a very small direct image readout can use cached
VideoPrism features for several simple synthetic taxonomies. The 512-epoch
curve shows that additional training improves this fixed development task,
while a small best-to-final regression and a growing train/dev gap appear near
the endpoint. It also shows that the macro hides a substantial task disparity:
color and shape are near ceiling while count remains much harder. These runs
do not identify a fixed capacity limit.

The 30,760-parameter image readout is not the conditional AV gate. The earlier
gate accounting remains: the gate MLP itself is 61,634 parameters and the full
additional AV residual path is 843,075 parameters. Their parameter/weight
overhead is small beside a 114M image encoder, but gate runtime latency and
memory have not been measured. More importantly, this image-only result does
not provide synchronized, complementary AV supervision and does not justify
training or adopting an AV gate. R2-M's both-required task, unimodal controls,
and intervention suite remain open.
