# PE-Core CLEVR-4 frozen readout result (2026-10-21)

## Run record

The preregistered development probe in
[`R2_PE_CORE_CLEVR4_READOUT_PROTOCOL_20261021.md`](R2_PE_CORE_CLEVR4_READOUT_PROTOCOL_20261021.md)
completed on the local RTX 3080 Laptop. It used seed 17, 8 epochs, 8,424
training images, the fixed 512-image checkpoint-selection split, and the
previously observed 1,595-image reporting subset. Train, selection, and reporting
IDs were disjoint; train/validation media SHA-256 overlap was zero. No sealed
audit was loaded.

- Start/end: 2026-10-21 05:05:44–05:08:39 UTC.
- Git HEAD at start: `fde8f873cafde64ddd1b98e608102ebd1df6188c`.
- The runner was uncommitted at launch; its executed-file SHA-256, retained in
  run metadata, is `94ab3a03b26d4353d0189c182aeb9df6d70ea06f6bcd18c657175452bcb9482d`.
- Effective config SHA-256: `6f4bcb004d88775d088a172554ce5659d80cdb554904c18b01f8cdff891f7104`.
- Encoder: PE-Core T16 at revision `7fe539ed578ac49a1c2b4f946e4b0747704c825a`,
  weight SHA-256 `477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`.
- All 69,546,113 encoder parameters stayed frozen. The only learned component
  was the shared 131,329-parameter candidate scorer. Combined pre-quantization
  count for this probe is 69,677,442 parameters.
- GPU work ran through the cooperative HotUGPU gate. It waited while external
  PID 17812 used the GPU, then reserved 3,200 MiB and launched only after the
  external process yielded. Peak child allocation was 423,511,040 bytes; GPU
  was released when the child finished. Feature extraction plus fit took
  170.44 seconds. Cloud cost: $0.
- Runtime: Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4, OpenCLIP 3.3.0.

## Learning curve

| Epoch | Train CE | Selection validation Accuracy | Selection validation NLL |
|---:|---:|---:|---:|
| 1 | 2.0657 | 41.41% | 1.6177 |
| 2 | 1.3295 | 63.38% | 1.0970 |
| 3 | 0.9817 | 69.97% | 0.8810 |
| 4 | 0.8169 | 73.68% | 0.7587 |
| 5 | 0.7164 | 75.05% | 0.6760 |
| 6 | 0.6373 | 76.81% | 0.6118 |
| 7 | 0.5756 | 77.59% | 0.5625 |
| 8 | 0.5296 | 79.30% | 0.5188 |

Selection NLL improved at every epoch, so the fixed rule selected epoch 8. Within
this eight-epoch budget, validation did not show the specified loss-based
overfit signal. This is a single seed on one synthetic development task; it does
not demonstrate a plateau or generalization beyond this benchmark.

## One-time reporting result

| Task | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Color | 98.75% | 0.0573 | 0.0229 | 0.0288 | 0.9591 |
| Texture | 93.10% | 0.2467 | 0.1142 | 0.0676 | 0.8635 |
| Shape | 98.68% | 0.0551 | 0.0254 | 0.0216 | 0.9656 |
| Count | 29.28% | 1.7026 | 0.7784 | 0.0769 | 0.2479 |
| Macro | 79.95% | 0.5154 | 0.2352 | 0.0432 | 0.7590 |

The learned readout improved macro Accuracy by 37.15 percentage points over the
fixed zero-shot PE-Core score (42.81%), but remained 8.10 points below the
historical Path A macro Accuracy of 88.06% on these same reporting IDs/options.
Against Path A, this candidate was −0.69 points on color, +1.25 on texture,
−1.07 on shape, and −31.91 on count. Its macro NLL/Brier/ECE were also worse
than Path A's 0.2912/0.1536/0.0177. These are development comparisons, not a
blind evaluation; the model paths also differ in encoder/readout design. They do
not establish causal superiority of any one architecture.

The count result is the main bottleneck. This task tests visual enumeration over
ten supplied choices; a strong text/image similarity embedding plus a small
candidate MLP did not recover it. The result does show that a compact learned
Decision scorer can extract much more useful performance from frozen PE-Core
features than direct zero-shot similarity on this dataset. It does not show that
PE-Core can meet the project's multimodal quality target, support video/audio,
or generalize to natural images. No product promotion follows.

## Preserved artifacts

All artifacts remain outside Git at
`C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-04-clevr4-readout`.
Run status is `complete`; best epoch 8 was reloaded and selection predictions
matched exactly. The directory retains the best and last-epoch readouts,
FP16 frozen feature cache, eight-epoch JSONL history, selection and reporting
probabilities, effective config, and run metadata. Key hashes:

- `run-report.json`: `494a7b4b9cbfd6306534152a21c915d654df9ad8d57099afc3bc7642be26f73e`
- `run-metadata.json`: `8bdf9ddb4eb69d3a024e4690db1f4558524951e0ef9a715f99e29a82d1728fdd`
- `best-readout.pt`: `261ceae2e75fca2d2d6b9a03ef4106783261fcc9ae7306af0b98809aef5b6e1d`
- `frozen-features.npz`: `c90307c024e7a60f3208230dcac366b5b9ec9cf81e00ece2926bcdf7a86848eb`
- `training-history.jsonl`: `7199501f9d341e8bc93d552e30eb17365b36d2d259e83ea8328523dc8c9e455c`
