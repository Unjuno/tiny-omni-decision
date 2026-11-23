# R2 Path A LINE DistilBERT Common Fit — 2026-11-24

## Decision

The fixed BERT text-feature integration completed its 2,048-update run and
passed exact best/final checkpoint replay. It did **not** improve the common
four-modality development score over the matched native baseline. Keep it as a
reproducible research result; do not promote it as the product checkpoint or
start an unstructured BERT sweep. The validation corpus is already exposed, so
all comparisons here are development evidence, not blind confirmation.

## Pinned setup

- Source commit: `77c2ffa3758c1e9145833d9125d790d92afca951` on
  `codex/r2-patha-bert-text-tower-20261124`.
- Model: `line-corporation/line-distilbert-base-japanese`, revision
  `c32d05753d0d4be0f60482291a9452ec7e6f7d68`, Apache-2.0. The six-layer
  checkpoint SHA-256 is
  `9d8b678bd9ce14363d5e6e3dc53c7dd7c76f8ad11394ccffd200f15e9f6819eb`;
  the frozen four-layer feature tower has 53,912,064 parameters.
- Effective config: `configs/pretrained_reuse/path_a_line_distilbert_fit_v1.json`,
  SHA-256 `981b3abc1b8d317bf2db4ea25b59bb566965dec8070796d1a258f9d46ede27b5`.
  Seed 17; 2,048 updates; four examples/update; AdamW, LR 0.001, 62-update
  warmup, cosine schedule, weight decay 0.01, clip norm 1.0; validation every
  256 updates; selector is macro NLL. The native encoders are frozen. Only the
  1,184,641-parameter readout/lens is trained.
- Frozen BERT feature-cache receipt SHA-256:
  `0d30d3a05b43918456233044229075a234f685527b31590c9609818bbc9d6f84`.
- Frozen common-corpus receipt SHA-256:
  `47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49`;
  validation JSONL SHA-256:
  `1f545a6e8d10b2712980f9ada58a0e5eef5b21021e4e01d6483965889a970326`.
- The BERT and native baseline validation ID/order files are byte-identical:
  SHA-256 `DB7D0F1D09319A48A7725ED36CCB515CDCC85A409F019400CF0062BE38768F30`
  (24,346 examples).
- Initialization was the original seed-17 native readout plus a newly seeded
  768-to-384 BERT text lens. It was not a warm start from another BERT fit.
  The baseline's selected checkpoint was its own validation-selected step
  1,792; the BERT candidate selected step 1,536.
- Parameter accounting is an arithmetic composition, not an assembled-runtime
  measurement: replacing the 70,830,722-parameter DeBERTa entry with the
  53,912,064-parameter four-layer LINE encoder gives 178,446,871 frozen
  parameters plus 1,184,641 readout/lens parameters = **179,631,512** total
  pre-quantization parameters. Package size and target-device inference remain
  unmeasured. No quantization was performed.

## Execution and artifacts

Training command:

```powershell
python -m scripts.train_native_common --config configs/pretrained_reuse/path_a_line_distilbert_fit_v1.json --phase train --variant baseline --output C:\CodexArtifacts\pretrained-reuse-r2-local\path-a-line-distilbert-common-fit-20261124-attempt-02 --reference C:\CodexArtifacts\pretrained-reuse-r2-local\path-a-line-distilbert-common-smoke-20261124-attempt-03 --reference-sha256 1b80a961252eb7fc7fdb3bf9bc89c4231ad65bdadc2831630759ef3524fcba95
```

- Run status: `PASS_NATIVE_FIT_TRAIN`; step 2,048; best step 1,536. The run
  started `2026-11-24 07:08:33 +09:00` and finished `07:20:26 +09:00`;
  wall time 713.3 s. Training took 64.7 s and validation evaluation 535.0 s.
- Local hardware/runtime: RTX 3080 Laptop GPU, 16 GiB; Python 3.11.9,
  PyTorch 2.6.0+cu124, CUDA 12.4, float32, TF32 disabled. Peak allocated VRAM
  257,305,600 bytes; peak reserved VRAM 362,807,296 bytes; peak process working
  set 3,978,866,688 bytes. Cloud cost: $0.
- The selected checkpoint represents 6,144 consumed examples (5,120 unique
  questions, 1,024 repeats, 2,466 unique media assets, 2,744 groups). The full
  run consumed 8,192 examples (6,656 unique questions, 1,536 repeats, 3,114
  unique media assets, 3,382 groups). Each modality contributed 2,048 consumed
  examples to the full run. This accounting is from the saved sampler receipt.
- Fit result SHA-256:
  `60852e971abb1089d0957badd7b94d22fe32b11aa1952745d43c49f64c9854c8`.
  Best checkpoint SHA-256:
  `c24590cd9c7fd50d3cfe45715f8ae44224b8420947a093950128abb3ed60b28f`;
  final checkpoint SHA-256:
  `929a7777f7cf9b42ed0908c0b4201ad428afe49802889336733d14518cab8678`.
- Replay command used `--phase replay` in a new attempt-01 output directory,
  pinned to the fit receipt. Status `PASS_NATIVE_FIT_REPLAY`; best and final
  checkpoints were reloaded and their validation and train-probe score arrays
  exactly matched the saved arrays (`replay_max_abs=0`). Replay receipt SHA-256:
  `862059d21741e563871f2ebee10cf52114776b4ceccd929a3a724c3e9fd3f17db`.
- Sealed audit and calibration data were not loaded. Existing baseline data,
  checkpoints, feature banks, and other worktrees were read-only. Checkpoints
  remain outside Git.

## Learning curve

All validation measures below use the same 24,346 IDs. Accuracy columns are
per-modality; macro measures average the four modality-level scores.

| Step | Macro accuracy | Macro NLL | Macro Brier | Macro ECE | Audio acc. | Image acc. | Text acc. | Video acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.215 | 2.221 | 0.798 | 0.070 | 0.501 | 0.000 | 0.174 | 0.230 |
| 256 | 0.276 | 1.869 | 0.757 | 0.066 | 0.525 | 0.096 | 0.188 | 0.335 |
| 512 | 0.273 | 1.809 | 0.751 | 0.061 | 0.527 | 0.096 | 0.200 | 0.310 |
| 768 | 0.281 | 1.796 | 0.753 | 0.070 | 0.527 | 0.105 | 0.189 | 0.343 |
| 1,024 | 0.286 | 1.768 | 0.747 | 0.068 | 0.526 | 0.102 | 0.192 | 0.365 |
| 1,280 | 0.290 | 1.777 | 0.748 | 0.047 | 0.530 | 0.119 | 0.192 | 0.355 |
| 1,536 (selected) | 0.297 | 1.738 | 0.746 | 0.068 | 0.547 | 0.113 | 0.196 | 0.370 |
| 1,792 | 0.301 | 1.742 | 0.746 | 0.071 | 0.543 | 0.132 | 0.197 | 0.368 |
| 2,048 (final) | 0.296 | 1.741 | 0.746 | 0.066 | 0.549 | 0.133 | 0.198 | 0.343 |

## Matched selected-checkpoint comparison

Both selected checkpoints use the same option-ordering, preprocessing, corpus,
validation IDs, and metrics. Macro values are averages over modality-level
values, rather than sample-weighted aggregates.

| Measure | Native baseline, step 1,792 | BERT text lens, step 1,536 |
|---|---:|---:|
| Macro accuracy | 0.3071 | 0.2973 |
| Macro NLL | 1.6865 | 1.7378 |
| Macro Brier | 0.7402 | 0.7458 |
| Macro ECE | 0.0842 | 0.0677 |

| Modality (n) | Baseline acc / NLL / Brier / ECE | BERT acc / NLL / Brier / ECE |
|---|---|---|
| Audio (11,166) | 0.5428 / 1.1467 / 0.5364 / 0.0707 | 0.5466 / 1.1642 / 0.5433 / 0.0864 |
| Image (2,048) | 0.2012 / 2.1651 / 0.8652 / 0.0311 | 0.1133 / 2.2868 / 0.8970 / 0.0284 |
| Text (10,732) | 0.2089 / 1.9211 / 0.8237 / 0.0742 | 0.1961 / 1.9856 / 0.8079 / 0.0238 |
| Video (400) | 0.3150 / 1.3393 / 0.7006 / 0.0842 | 0.3700 / 1.3410 / 0.7000 / 0.0682 |

The BERT integration raises Video accuracy by 5.5 percentage points and Audio
by 0.4 points, but lowers Image by 8.8 points and Text by 1.3 points. Macro
accuracy falls by 1.0 point and macro NLL worsens by 0.0513. The lower macro
ECE does not offset these quality losses. This is a measured cross-modality
trade-off, not evidence that all BERT-family models are unsuitable.

## Limits and next gate

The run proves a reproducible frozen BERT-feature integration can train inside
the provisional parameter arithmetic and that checkpoint selection/reload is
sound. It does not reach the product's 95–98% aspiration, establish quality on
independent data, demonstrate a packaged runtime below 200 MB, or validate
synchronized multimodal complementarity. The common development validation
has been reused in prior experiments and must remain labeled exposed.

Do not continue tuning this exposed validation or promote this candidate. Keep
the native Path A checkpoint as the better matched common baseline. The next
work should use a fresh, admitted, group-disjoint task/validation pair and a
predeclared diagnostic that addresses the observed image/text degradation;
the current result alone does not justify a shared-transformer or weight-merging
experiment. R0–R8 and the four-modality product gates remain open.
