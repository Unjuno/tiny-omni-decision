# FLAN-T5-small: actual pinned runtime reference

Follow-up to the metadata-only budget audit. This is a research reference, not a
drop-in upgrade of the current Omni stack or an adopted alternative architecture.
Existing VideoPrism + Whisper + this model still exceed200M before fusion.

Downloaded revision0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab into a fresh directory.
307,867,048-byte safetensors SHA256 matches the previously pinned publisher LFS
value495fa51e204676f1a857a9fc13c4c89f3f5ba9f480b898cebca02add25e6d749.
Tokenizer/config files and individual hashes are retained in local download.json.
The prior Apache2.0 card audit applies to this exact revision.

Installed Transformers T5Config.__post_init__ deliberately forces
tie_word_embeddings=True and derives scale_decoder_outputs from the incoming flag.
The model class then ties lm_head to shared as well as its two input embeddings.
That does not match this checkpoint's distinct shared.weight and lm_head.weight.

A local subclass restricts _tied_weights_keys to encoder/decoder input embeddings.
No installed library or old model is changed. Build with the published config,
verify scale_decoder_outputs=False, and fill the two omitted input-alias state keys
from shared.weight before strict load. All resident tensors exactly match the
checkpoint, including the separate output matrix. Actual unique parameters:
**76,961,152**. Inputs share storage; output does not.

CPU FP32,4threads: two short prompts yield finite [2,1,32128] logits; repeat and
fresh-instance strict reload are exactly equal. Timed build/forward/reload portion
1.558844seconds, excluding download and checkpoint deserialization. No GPU use,
training, calibration, benchmark scoring, or sealed access. Two smoke prompts
(water wet; larger of2/7) have top first tokens “no” and “2”; do not treat this
as useful reasoning performance, nor as a statistically meaningful quality test.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r1/flan-t5-small-runtime-01/.
Harness: C:/CodexArtifacts/pretrained-reuse-r1/audit_flan_untied_runtime.py,
SHA2566286765156f246f915604e7260e62699f52477a2f4b7c94776f34d87f191658f.
The initial inline subclass probe failed because installed Transformers opens its
class source file and cannot open <stdin>; a saved script resolved that runtime
issue. No checkpoint compatibility checks were bypassed.

This resolves local load feasibility only. Neither this smoke nor arithmetic
with smaller towers establishes a four-modality candidate. Any quality comparison
must freeze scoring and tasks first and retain earlier negative text results;
no prompt/seed search or adoption is justified by the current evidence.
