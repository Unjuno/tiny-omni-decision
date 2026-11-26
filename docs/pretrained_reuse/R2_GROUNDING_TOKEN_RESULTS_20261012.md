# Descriptive token readout: completed, no product adoption

Source `3dc44f7`. All three modes completed24epochs on the frozen1,337-question /
256-scene training and2,200-question /200-scene development validation set.
Every one of456 re-extracted token tensors matched the prior SHA-256 and pooled
feature within1e-5. No tower trained. Full variable candidate sets retained.
Separate two-epoch smoke passed first; all best reloads are exact.

| Selected mode | Epoch | Macro Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Attention | 3 | 0.39993 | 1.21535 | 0.65750 | 0.06589 |
| Mean | 2 | 0.38907 | 1.22064 | 0.65700 | 0.03471 |
| No video | 2 | 0.35835 | 1.23764 | 0.66926 | 0.05419 |

Metrics weight five taxonomies equally. Paired scene bootstrap:10,000 draws,
seed17, percentile95% intervals; validation selection and repeated use remain
limitations. Attention-minus-mean Accuracy+1.086pp [-1.663,+3.869] and
NLL-0.00529 [-0.02368,+0.01259] do not establish an attention advantage.
Attention-minus-no-video Accuracy+4.159pp [+1.305,+7.049] and NLL-0.02229
[-0.04208,-0.00261] support limited scene information in this development task.
Mean also improves over no-video by+3.073pp [+0.956,+5.152].

Matched-minus-mismatched video Accuracy: attention+5.604pp [+3.568,+7.734],
mean+4.174pp [+1.839,+6.552]. Corresponding NLL differences are-0.07157 and
-0.03175, both intervals below zero. These are scene interventions, not temporal
order tests. No-video matched/shuffled predictions are exactly identical.

## Overfitting and bounded decision

Attention train NLL falls1.09793->0.24034 from selected epoch3 to final24,
while validation NLL rises1.21535->2.34501. Final train Accuracy91.72% versus
validation40.55% is clear overfitting, not product success. Mean also overfits:
train NLL1.13248->0.56262, validation1.22064->1.53140. No-video selected2 is retained.
No arbitrary extension, extra seed, capacity increase or product adoption.
The new modality-native R2-M roadmap now takes priority over this text-conditioned
probe family. Preserve these results as optional-criterion reference controls.

## Resource and artifact accounting

Each mode stores393,984 head parameters; mean/no-video key/position weights are
unused, so active capacity differs. Frozen towers total131,755,008; with one head
132,148,992 parameters. This is not an assembled full Omni runtime.
Each mode consumed32,088 encounters:1,337 unique questions and30,751 repeats.
Selected attention consumed4,011, mean/no-video2,674. All256 train scenes remain.

CPU token extraction plus frozen text preparation867.52s; attention58.43s,
mean36.76s, no-video37.98s. Total measured intervals1000.69s, excluding some setup.
Peak allocated VRAM2,980,033,536bytes includes the token bank. Token extraction
used pinned CPU JAX, training local RTX3080 Laptop. No cloud cost; electricity
unmeasured. All per-example predictions, histories, environment-bearing worker
log, input hashes and best/final hashes are retained.

Run: `C:/CodexArtifacts/pretrained-reuse-r2/grounding-token-training-01`.
Command: `python -m scripts.train_grounding_tokens --output C:/CodexArtifacts/pretrained-reuse-r2/grounding-token-training-01`.
Do not rerun into existing output. [Complete report](R2_GROUNDING_TOKEN_RESULTS_20261012.json).
Old data/models/checkpoints remain unchanged; no sealed audit was accessed.
