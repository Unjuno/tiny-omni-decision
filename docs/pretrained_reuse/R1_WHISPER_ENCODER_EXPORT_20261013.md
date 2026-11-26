# Frozen Whisper encoder reuse: actual compact export

Verified the existing pinned Whisper-tiny checkpoint and exported only its native
Encoder using Transformers. Original full37,760,640parameters remain unchanged;
new Encoder has8,208,384parameters, saving29,552,256relative to the full model.
New safetensors32,839,984bytes,SHA
`d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
All registered positional parameters are included, not just trainable weights.

Strict encoder-state loading and an independent exported-weight reload passed.
Four complete training-assigned Lombard Grid audio samples passed finite,
bitwise repeat, original-Encoder parity and exported-reload parity. FP32 CPU,
4threads,Torch2.6.0+cu124,1.82s for feature comparisons excluding load/export.
No learning, GPU job, transcription generation or model-quality evaluation.

Official frontend uses16kHz log-Mel and30-second padding. Retain the first
`ceil(samples/(2*hop_length))`384D steps corresponding to the physical clip;
save ordered features. No source transcript, filename or gold word alignment is
an Encoder input. This masking is not a measured word alignment or causal guarantee.
Future corpus extraction must reject oversize clips rather than silently truncate.

The Encoder alone does **not** preserve the full model's ASR/candidate-transcript
Decoder behavior; that existing runtime is retained. A native Decision readout
still needs training/evaluation. No95–98% claim or complete Omni promotion follows.
Known separate component counts (VideoPrism, this Encoder, EfficientAT, MiniLM)
sum141,924,631before heads/lenses; this is budget arithmetic, not an assembled
and quality-qualified product. Future exports must count every extra component.

## License and artifacts

Reference `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
The [pinned HF card](https://huggingface.co/openai/whisper-tiny/blob/169d4a4341b33bc18d8881c4b69c2e104e1cc0af/README.md)
declares Apache2.0; the [original OpenAI implementation](https://github.com/openai/whisper#license)
describes MIT code/model weights. These scopes are recorded distinctly.
The initial local runtime report inaccurately named an MIT checkpoint audit;
an explicit `license-addendum.json` corrects this while preserving the original
report. Saved pinned card and Apache license text/notices accompany the export.
The first notice check assumed license text was within500bytes; the successful
check parses the complete bounded YAML frontmatter. No rights gate was relaxed.
Exact pretraining overlap with this public speech corpus remains UNKNOWN.

Local `C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01/` holds
new weights/config/frontend, ordered token smokes, source snapshot, hashes and
license addendum. Source runner:
`C:/CodexArtifacts/pretrained-reuse-r2/lombard_grid_whisper_encoder_smoke.py`.
All prior checkpoints/configs/data are untouched. Cloud cost0.
