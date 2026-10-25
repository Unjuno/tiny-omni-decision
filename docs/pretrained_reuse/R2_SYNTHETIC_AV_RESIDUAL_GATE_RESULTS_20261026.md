# Identity-initialized residual gate on raw AV features (2026-10-26)

## Question and scope

Two projected/gated temporal readouts failed on the fixed synthetic AV task,
while the direct-concatenation LSTM solved it. This experiment tests a minimal
gate that keeps the reference LSTM's raw audio/video/query input intact. A
query-conditioned per-step gate adds bounded residual gains to each raw
modality. Its initialization exactly reproduces the original LSTM logits and
weights for seed 17.

This is synthetic architecture evidence only. The validation set was used for
checkpoint selection and is not blind. It does not establish natural AV
transfer or the product's 95–98% objective. No sealed audit was opened. All
older runs, data, and checkpoints remained read-only.

## Frozen setup and implementation

- Model input to the LSTM remains the raw 384-wide audio feature, raw
  768-wide video feature, and 2-wide query at each of eight steps. A small
  per-step gate receives LayerNorm-projected audio, video, and query, and emits
  two softmax weights. Each raw feature stream is scaled by
  `1 + (weight - 0.5)`, bounded to `[0.5, 1.5]`; neither stream can be zeroed.
- The gate's final layer is zero-initialized. The LSTM and classifier are
  instantiated first in the same order as the reference, so with seed 17 the
  initial LSTM and classifier weights match exactly. Unit tests confirm that
  the initial logits are bit-for-bit equal to the original temporal LSTM.
- Parameters: 595,844 total trainable readout parameters, 7.27% above the
  reference LSTM and within the 10% budget. Offline feature towers plus head
  total 116,922,267 parameters; this remains a research readout, not a product
  export.
- Data/encoders/features: unchanged synthetic task, VideoPrism and EfficientAT
  caches, 4,096 train rows / 512 groups, 1,024 validation rows / 128 groups.
  Validation IDs/order and sampler stream exactly match the earlier runs.
  Manifest SHA-256:
  `19dabb8935628d90ebd1b843148ed174ee49a931c4ca61b687edc415b67cde5c`;
  feature cache SHA-256:
  `0685402a5a5d4603f8c9f9dabaae7da94f5ef6ab0cc38fa76323385f1e82db16`;
  permutation SHA-256:
  `7988a28d484c35ca071cc07a57e9f8c509860e28c175aa7e49d097ee0afe60e9`.
- Seed 17; AdamW, LR 0.001, weight decay 0.0001, batch 32, 128 epochs,
  validation every four epochs. Best selected by minimum validation NLL,
  strict improvement only, earliest exact tie retained. Best and terminal are
  separate, although the terminal step also improved the selector.
- Full run: 16,384 optimizer updates / 524,288 row presentations. At every
  epoch, the 4,096 unique training rows are traversed once; this is repeated
  training, not 524,288 unique records. Best step is 16,384, so best and
  terminal coincide by selector result, with both files retained.

## Learning curve

This candidate was at chance at step 512, rose sharply before step 2,048, and
then maintained near-perfect validation Accuracy through later updates. The
previously failing gated architectures stayed at chance for their entire
16,384-step runs.

| Step | Validation Accuracy | NLL | Brier | ECE | Online train CE | Online train Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.5000 | 0.693156 | 0.500008 | 0.002077 | 0.693594 | 0.4954 |
| 2,048 | 0.9990 | 0.011397 | 0.001888 | 0.010767 | 0.092910 | 0.9773 |
| 3,584 | 1.0000 | 0.004148 | 0.000544 | 0.004001 | 0.025192 | 0.9917 |
| 4,096 | 1.0000 | 0.001179 | 0.000367 | 0.001056 | 0.016398 | 0.9932 |
| 4,608 | 0.9971 | 0.006956 | 0.003178 | 0.005121 | 0.010915 | 0.9963 |
| 8,192 | 1.0000 | 0.001114 | 0.000543 | 0.000942 | 0.005314 | 0.9963 |
| 11,264 | 0.9990 | 0.003524 | 0.001709 | 0.001833 | 0.010424 | 0.9958 |
| 15,360 | 0.9990 | 0.001567 | 0.000943 | 0.001208 | 0.003228 | 0.9978 |
| 16,384 (best/terminal) | 1.0000 | 0.000645 | 0.000240 | 0.000572 | 0.003370 | 0.9978 |

Online train metrics are changing-weight epoch summaries, not fixed-checkpoint
generalization gaps. A separate post-selection evaluation of the best weights
on the fixed train set gave 0.9978 Accuracy / 0.003658 NLL / 0.002598 Brier /
0.001216 ECE, versus validation 1.0000 / 0.000645 / 0.000240 / 0.000572.
There is no obvious overfit signal in these measures, but the generated task is
small and shares the four symbol/tone prototypes across splits by design.

## Same-validation comparison and input controls

The five readout architectures across four run receipts' best/terminal
checkpoints were loaded by one evaluator using the same cache and 1,024
validation IDs.

| Readout / best checkpoint | Trainable parameters | Step | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original temporal LSTM | 555,458 | 11,264 | 1.0000 | 0.000585 | 0.000278 | 0.000493 | 0.999507 |
| Bidirectional conditional AV | 513,412 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000022 | 0.500022 |
| Query-conditioned temporal AV | 528,388 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |
| Bypass-conditioned temporal AV | 521,476 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000106 | 0.500106 |
| Residual-gated raw temporal AV (this run) | 595,844 | 16,384 | 1.0000 | 0.000645 | 0.000240 | 0.000572 | 0.999428 |

The paired group bootstrap (10,000 draws, seed 17, 128 validation groups)
estimates the candidate-minus-LSTM Accuracy difference as 0.0000 with interval
`[0.0000, 0.0000]`. It is a descriptive interval on validation used to select
the checkpoints. The candidate's point Accuracy matches the LSTM; its NLL is
slightly higher, Brier slightly lower, and ECE slightly higher.

| Candidate input condition at best checkpoint | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Full aligned AV | 1.0000 | 0.000645 | 0.000240 | 0.000572 |
| Audio only | 0.5000 | 6.550655 | 0.998388 | 0.499192 |
| Video only | 0.5000 | 8.147302 | 1.000000 | 0.500000 |
| Both missing | 0.5000 | 6.827808 | 0.999971 | 0.499986 |
| Audio shuffled within group/query | 0.5000 | 7.034013 | 1.000092 | 0.500376 |
| Reverse audio time only | 0.5146 | 8.113479 | 0.967849 | 0.484483 |
| Reverse both streams | 0.9990 | 0.005880 | 0.002144 | 0.001152 |

Audio-only, video-only, missing-input, and mispaired-audio controls fall to
chance; the aligned streams are needed. Reversing both streams preserves the
sequence comparison rule and retains high performance; reversing only audio
breaks alignment and returns to chance. This is evidence of both-stream
dependence for this generated task, not natural AV reasoning.

The learned per-step gate is dynamic rather than globally collapsed: mean
weights were `[0.3886, 0.6114]` for `[audio, video]` on validation (train
`[0.3878, 0.6122]`), with per-time-step standard deviation about 0.469. The
residual gain bounds ensure these weights cannot erase the raw channels.

## Compute and artifact hashes

- GPU: NVIDIA GeForce RTX 3080 Laptop 16 GiB; Python 3.11.9; PyTorch
  2.6.0+cu124; CUDA 12.4.
- Training ran behind the cooperative HotUGPU reservation. Runtime 120.70 s;
  peak allocated/reserved memory 508,658,176 / 562,036,736 bytes. CPU
  evaluation of four runs' checkpoints and controls: 54.24 s. Cloud cost: $0.
- CPU smoke completed 2,048 smoke updates and best reload verification with
  zero logit error; it is separate from the full run.
- Source commit `0f471a9c3109b09ab322e8c34764ebf999917623`; config SHA-256
  `ed06d6d6220f2759a1bb978c3327adff8645fc043fe4e2971989b699aa90cb49`.
- Training receipt SHA-256:
  `895a2696710b161462288f13316c8814e3606151a4893a9611ff2f6792344afc`;
  history:
  `fbfa2b27bffb451465e22615092cf484f8ffca713b37121ce27cab4017fa75a9`;
  best weights:
  `1cc33b123cea02c5e57f8cd27f69fe78d1ab459d18e3a228a8f3c6a504a0f811`;
  terminal weights:
  `e862cdcd8e7b05b8cca1f065acbd536438e43d0503be89f361925fb31d435025`.
- Evaluation receipt SHA-256:
  `735816afda33eec2f6678e1ed3d52ce517aff5cc8e5f7ed32072f2a81108fee6`;
  saved predictions:
  `fd4af1ef5c481134367dac89236da6d448a0ba89281cdae8eaa0676374b4039a`.
- External outputs:
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\synthetic-av-residual-gated-temporal-v4-attempt-01`
  and
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\synthetic-av-residual-gated-temporal-v4-eval-attempt-01`.
  No media/checkpoint files were added to Git.

## Interpretation

The direct raw-feature recurrent path is doing the essential work on this
generated task. The bounded identity-initialized gate adds learned conditional
modulation without hurting synthetic Accuracy, but the original LSTM already
reaches 100%; this run does not establish a gate benefit. The gate weights are
not evidence of useful general-purpose modality arbitration by themselves.

This is a validated synthetic-mechanics result only. Keep it separate from
product selection. The next gate is a rights-cleared, naturally recorded AV
task with independent source groups; no natural AV source has yet cleared that
benchmark requirement. Do not promote this head or claim 95–98% multimodal
quality from these synthetic results.
