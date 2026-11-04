# Text-free audio Decision probe — 2026-10-17

## Question and scope

This one-shot Path A probe asks whether a small Decision head can classify a
fixed English command taxonomy from frozen audio features without a text encoder
or candidate-text input. It is a narrow M-0/M-1 implementation milestone, not a
general audio, multimodal, or product-quality result. The existing split and
features are reused unchanged; validation has previously selected the Audio+Text
readout, so this is a matched development comparison and not blind evidence.
No sealed audit, raw-media re-encoding, GPU, or existing model/run mutation was
used.

## Frozen inputs and implementation

- Speech Commands v0.02, pinned at
  `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`.
- 1,280 train examples / 829 speaker groups; 640 validation examples / 206
  speaker groups; balanced over `yes/no/up/down/left/right/on/off/stop/go`.
- Train/validation speaker intersection: 0. Audio SHA intersection: 0. The
  runner rechecked every media path and SHA against the frozen JSONL manifests.
- Frozen feature source: Whisper Tiny encoder at
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; feature cache SHA-256
  `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`.
- Input: one mean-pooled 384-wide audio vector. The Decision head receives no
  question, transcript, candidate text, text embedding, or text encoder.
- Readout: one linear 384→10 layer, 3,850 trainable parameters. Eight epochs,
  seed 17, batch size 64, AdamW at `1e-3`, weight decay `1e-4`; best selected by
  minimum validation NLL. 160 optimizer updates, CPU training time 0.19 s.
- Outputs are typed `option_id` / `class_id` JSON probabilities. Candidate order
  and a supplied two-class subset are covered by CPU tests. Save/reload logits
  matched exactly.
- The Whisper encoder contains 8,208,384 parameters in its safetensors tensors
  (32,833,536 FP32 tensor bytes). The on-disk checkpoint is 151,061,672 bytes
  and includes unused decoder and tokenizer assets; this probe did not create a
  deployable pruned encoder package or measure end-to-end runtime.

## Results on the same reused validation IDs

| Readout | Accuracy | NLL | Brier | ECE-15 | Readout trainable params |
|---|---:|---:|---:|---:|---:|
| Existing audio + MiniLM candidate scorer, selected epoch 8 | 0.9531 | 0.1624 | 0.0661 | 0.0377 | 196,865 |
| New text-free linear audio readout, selected epoch 8 | 0.9656 | 0.2834 | 0.0986 | 0.1515 | 3,850 |

The text-free head gains 1.25 percentage points of Accuracy on this view, while
NLL worsens by 0.121, Brier by 0.0326, and ECE by 0.1138. It is therefore more
accurate but materially less calibrated. Train/validation Accuracy is 0.9688 /
0.9656; no train/validation Accuracy gap is apparent in this small closed task.
Validation NLL continued decreasing through epoch 8, so this budget did not
show a clear plateau.

This does not establish open-vocabulary speech, transcript understanding,
Japanese audio, noisy or OOD command performance, or complementary audio/video
reasoning. Since the validation set was already used for previous checkpoint
selection, the comparison must not be presented as independent or blind.

## Reproducibility

Code:

- `src/tiny_omni_decision/textfree_audio.py`
- `scripts/train_textfree_audio_probe.py`
- `tests/test_textfree_audio.py`

Reproduction command (from repository root, in the pinned local environment):

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m scripts.train_textfree_audio_probe `
  --feature-cache C:\CodexArtifacts\pretrained-reuse-r2\audio-probe-v0-retry3\audio-features.pt `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
  --output-dir <new-empty-output-directory>
```

Canonical run receipt and immutable artifacts are external to Git:
`C:\CodexArtifacts\pretrained-reuse-r2\textfree-audio-decision-03\`.
Earlier same-seed runs 01 and 02 are preserved. Run 02 added config/source hashes
and explicit save/reload checking; run 03 reran the unchanged recipe after adding
stricter taxonomy and option-schema checks. All three runs produced the same
readout and prediction hashes. Run 03 reports
config SHA-256 `00c24b553ae74f01504a55d56ed1aec95b18b56edc3c00a16cd14e7657b3b72b`,
readout SHA-256 `46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb`,
and prediction SHA-256
`c286adc0e19ee8ba5ce9f3fc1da2939171227aae0e98da6ea3c26b859f099aba`.
The selected readout file is 16,886 bytes; the full run-report SHA-256 is
`a8452995e3d8d55f5a580177f9fd5251b392642ca7fe98396a70900b2d46ddcc`.

## AV gate cost clarification

The previously reported 843,075 parameters are the *entire added AV residual
path* (latent pooling, parallel exchange, candidate attention, gates and
correction scorer), not the gate alone. In the checked implementation, the
conditional gate MLP is 61,634 parameters: 0.031% of the 197,097,949-parameter
reference model, about 0.123 MB of FP16 weights. The complete added AV path is
about 0.43% of that model, about 1.69 MB of FP16 weights. These are arithmetic
weight-size estimates, not measured runtime memory or latency. Repeated baseline
projections and attention computation are the more relevant runtime costs.

Even this small gate remains untrained and unvalidated for quality: the frozen
common corpus contains zero paired audio+video observations. The text-free
command result cannot justify AV fusion. First admit a source with independently
grouped, synchronized, complementary audio/video labels; then compare against
matched unimodal controls. No product or 95–98% quality gate is closed.
