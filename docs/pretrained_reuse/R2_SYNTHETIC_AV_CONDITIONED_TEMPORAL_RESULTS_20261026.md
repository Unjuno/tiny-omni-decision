# Query-conditioned temporal AV readout result (2026-10-26)

## Question and scope

This experiment tested one simpler alternative to the prior bidirectional
cross-attention readout: at each aligned time step, a learned gate conditions
the audio/video mixture on the query, then a two-layer LSTM integrates the
fused sequence. It uses the same frozen synthetic AV feature cache, task,
split, seed, optimizer, batch, epoch budget, and checkpoint selector as the
previous temporal-LSTM / conditional-attention comparison.

This is a synthetic architecture diagnostic only. It is not natural-media
transfer, a general AV decision result, or evidence about the 95–98% product
objective. Validation was used for checkpoint selection; it is not a blind
test. No sealed audit was opened. The original run and artifacts were read-only.

## Implementation

- Model: per-step audio/video projections and LayerNorm; a query-conditioned
  softmax gate mixes the two modalities; the gated vector and projected query
  enter a two-layer, width-128 LSTM; the final state produces two logits.
- Trainable parameters: 528,388, 4.87% fewer than the original temporal LSTM
  (555,458) and within the predeclared 10% comparison envelope.
- Frozen inputs: VideoPrism `[N,8,768]`, EfficientAT DymN04-AS `[4,384]`
  expanded across the aligned eight event positions.
- Data: 4,096 train rows / 512 groups and 1,024 validation rows / 128 groups.
  Manifest SHA-256 `19dabb8935628d90ebd1b843148ed174ee49a931c4ca61b687edc415b67cde5c`;
  aligned feature-cache SHA-256
  `0685402a5a5d4603f8c9f9dabaae7da94f5ef6ab0cc38fa76323385f1e82db16`.
- Seed 17, AdamW, learning rate 0.001, weight decay 0.0001, batch 32,
  128 epochs, 128 updates per epoch. Validation every four epochs; minimum
  validation NLL with strict improvement and earliest tie retained. Terminal
  checkpoint is separate from best.
- Full run consumed 524,288 row presentations; best step 3,584 corresponds to
  114,688 presentations and had seen all 4,096 unique train rows. This is a
  cyclic multi-epoch synthetic task, not 524,288 distinct examples.
- No initialization was loaded from the previous learned heads. This was a
  fresh readout initialization on the frozen cached features.

## Result

The candidate did not learn the task. Validation and online train metrics
remained at chance throughout the full 16,384 updates. The only temporary
validation degradation (step 4,608) was not selected; training nevertheless
continued to the planned endpoint.

| Readout / selected checkpoint | Trainable params | Step | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Temporal LSTM (frozen reference run) | 555,458 | 11,264 | 1.0000 | 0.000585 | 0.000278 | 0.000493 | 0.999507 |
| Bidirectional conditional AV (frozen reference run) | 513,412 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000022 | 0.500022 |
| Query-conditioned temporal AV (this run) | 528,388 | 3,584 | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |

All rows above were recomputed with the same evaluator from saved checkpoints,
using the same 1,024 validation IDs and cached features. The selected-checkpoint
group-paired bootstrap (10,000 replicates, seed 17) estimates the candidate
minus LSTM accuracy difference as `-0.5000`, 95% percentile interval
`[-0.5000, -0.5000]` across 128 validation groups. The validation cohort also
selected checkpoints, so this interval is descriptive and optimistic; it is
not an independent-test interval.

| Candidate control at best checkpoint | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Full aligned AV | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |
| Audio only | 0.5000 | 0.693147 | 0.500000 | 0.000089 | 0.500089 |
| Video only | 0.5000 | 0.693147 | 0.500000 | 0.000271 | 0.500271 |
| Both missing | 0.5000 | 0.693147 | 0.500000 | 0.000278 | 0.500278 |
| Audio paired to another row in group | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |
| Reverse audio time | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |
| Reverse both time axes | 0.5000 | 0.693147 | 0.500000 | 0.000087 | 0.500087 |

The near-identical intervention outputs show that this candidate's selected
predictions did not use either input stream in a measurable way. This is
consistent with its chance-level online train accuracy and does not support a
claim that the gate learned useful modality selection.

## Post-selection optimization diagnostic

Without updating any weights, I ran one backward pass on the same 32 training
rows used as the first minibatch by both seed-17 runs. The SHA-256 of that
permutation-index list is
`e9cd0faea523039a7e46767c90987377abc5e7d01a90f559675d69c33b7f79dc`.
Raw feature RMS on this batch was 0.1542 for audio and 0.6116 for video; after
the candidate's projection and LayerNorm the per-element standard deviations
were similar (1.046 audio, 1.009 video).

| Selected readout | Audio projection grad L2 | Video projection grad L2 | Gate grad L2 | Temporal / attention grad L2 | Fixed-batch CE |
|---|---:|---:|---:|---:|---:|
| Query-conditioned temporal AV | 1.03e-5 | 2.14e-7 | 4.17e-7 | temporal LSTM: 1.67e-3 | 0.693157 |
| Bidirectional conditional AV | 1.53e-6 | 1.15e-5 | 1.37e-5 | audio-reads-video: 2.49e-5; video-reads-audio: 3.23e-5 | 0.693144 |

The query-conditioned gate assigned mean weights `[0.998432, 0.001568]` to
`[audio, video]` over all train time steps and `[0.998431, 0.001569]` over all
validation time steps. It had collapsed almost entirely to audio, and the
video projection gradient on the fixed batch was about 48 times smaller than
the audio projection gradient. Similar normalized feature scales suggest that
raw RMS alone does not explain this collapse. This post-selection diagnostic
on existing train/validation inputs was not used for checkpoint selection; it
narrowed the next architecture test but does not prove causality.

## Compute and reproducibility

- GPU: NVIDIA GeForce RTX 3080 Laptop, 16 GiB; Python 3.11.9; PyTorch
  2.6.0+cu124; CUDA 12.4.
- Training used the repository HotUGPU guard. It first detected a brief
  utilization race, released its reservation, then obtained a 2,418 MiB slot
  and launched only after the cooperative gate passed. No process was stopped.
- Training wall time: 124.01 s; peak allocated/reserved memory:
  451,748,864 / 465,567,744 bytes. CPU evaluation of all best/terminal
  checkpoints and controls: 33.21 s. Cloud cost: $0.
- CPU smoke passed separately (2,048 smoke updates, best reload error 0).
  It did not select the full-run checkpoint.
- A first guard invocation failed before reservation because direct script
  execution lacked the repository root on `PYTHONPATH`; no GPU training
  started. The successful invocation supplied the repo root in `PYTHONPATH`.
- Training source commit: `1425b66815aa122d8a8cf505aea6fbdc5c512398`;
  evaluation support commit: `d378bbe` (the evaluated code was the current
  tree with that change). Config SHA-256:
  `b81bc270895794a0449cd60cdd99ace7a2371623e7edd48bc3cd46ac39262664`.
- Successful run receipt SHA-256:
  `e2a050cf95c43d9b4d98da5121337b52b20afde2c0fe9d195508a6469fa535aa`.
  Best checkpoint SHA-256:
  `ddd7556f0befa7717915df15fe2ef90688281feb22442acf0eb2296f23eac049`;
  terminal checkpoint SHA-256:
  `05a7fbff4f0fe9a1a3a278222d0d3ad31683c044f47dc6b7114a2ca929655731`.
- Evaluation receipt SHA-256:
  `7771cf4b5b4c7c0fb275f81d3faa023a8802e30003358deb028e57714ea741fe`;
  saved prediction JSONL SHA-256:
  `1eee04360d54c4d16b363a677c212a4e30704cb2e2d4f0987df4691e6a4eb7ef`.
- Outputs are external at
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\synthetic-av-conditioned-temporal-v2-attempt-02`
  and
  `C:\\CodexArtifacts\\pretrained-reuse-r2\\synthetic-av-conditioned-temporal-v2-eval-attempt-02`.
  No weights or media were added to Git.

## Interpretation and next gate

This simpler query-conditioned temporal gate is **not supported** on this
synthetic task at the tested initialization, feature representation, and
optimization budget. It performed like the prior conditional-attention head,
while the original concatenated-input LSTM solved the rule. The post-selection
diagnostic points to a specific failure mode: the candidate gate suppresses
video and starves its video projection of gradient. A next, separate run may
test a residual/bypass path that preserves both projected modalities at the
LSTM input while retaining the gate as an auxiliary fusion signal. This is
narrower than increasing capacity or changing the encoders, and should retain
the same data, seed, and budget. It still would not justify xLSTM adoption or a
natural-media/product claim.
