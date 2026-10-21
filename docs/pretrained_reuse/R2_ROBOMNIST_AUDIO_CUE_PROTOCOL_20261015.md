# Fixed RoboMNIST acoustic collection-cue diagnostic

Purpose: inspect the three already admitted native WAVs from known TRAIN pilot
`Ar1_Vhigh_Ac0_Rp01_UNC062` for speech compatible with the public collection
counter. This is input/task admission evidence, not learned Decision accuracy,
speech ground truth, proof of the deployed collector, or a synchronized clock.
Source inspection already identified an espeak counter call, but its activity in
released media is UNKNOWN. Do not fit a joint head before investigating it.

## Immutable data/model and controls

Revalidate raw-input replay02 SHA
`5cd718a66f53d843ce30d9f1ba3519aa534ca977319491d90798367096576d51`
and all92 original pins. Use all three full mono44.1kHz PCM16 WAVs. No additional
media, position GT, observed final evaluation or sealed audit is opened.
Require all decoded PCM/metadata to match the original saved native arrays.

Use existing full `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
offline. Weights SHA
`7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`;
all11 existing model/tokenizer/processor files must match immutable reference
config SHA `897add18c70b5d67a0811ae6b2dc4391e124c8a3113e550847a1a59e8b38dac8`.
Verify the reference before parsing and independently require the weight SHA.
Full37,760,640 resident parameters frozen/eval; CPUfloat32/eager/2threads/seed17.
Released Hub card Apache-2.0 and original OpenAI MIT source terms remain distinct.
The data is RoboMNIST v1 CC BY4.0; upstream collection code is read-only format
evidence and is not adopted as licensed implementation.

Pin the source inspection receipt
`docs/pretrained_reuse/R2_ROBOMNIST_CLOCK_SOURCE_20261015.json` at SHA
`cdea6567d160a58df291ff18a9896bef9391c9b76719560ea27651bb9ba52868`
before parsing; require all six underlying text hashes. Do not execute those files.

Use existing PyAV/libswresample to convert each complete native mono float32
PCM44,100Hz to16,000Hz, using flt/mono output and resampler defaults; flush the
resampler. No amplitude normalization, denoising, clipping, filtering based on
recognized words, external resampler installation, crop, synthesized timing,
counter subtraction or duration equalization. Require actual output duration
within one16kHz sample of native duration; record exact lengths and PCM hashes.
Native WhisperFeatureExtractor rescale/log-mel/padding/normalization behavior is
kept from the pinned checkpoint. Save full mel tensors and valid attention masks.

For each receiver, in order1/2/3, run exactly two conditions:
1. complete resampled real waveform;
2. same-length float32 zeros through the same preprocessing and decoder.

No audio class label/name/file-length feature enters the model. Original native
duration is only preprocessing/mask metadata. Greedy English transcription,
one beam, no sampling, temperature0.0, max_new_tokens128, no timestamps.
Use return_dict_in_generate=True to retain complete generated sequences including
the exact English language/task prefix; the installed short-form tensor API
otherwise removes that prefix. Preserve full IDs rather than reconstructing it.
Disable compression/logprob/no-speech fallback thresholds; pass a copied fixed
generation config to each call. Preserve native suppression tokens. Save exact
generated IDs and raw decoded text; missing terminal EOS is an incomplete run.
No prompt/distractor/seed/model/decoder-setting search or truncation rescue.

Spelled/numeric0–20 observations may be recorded descriptively. Do not infer
correct speech, class leakage, target labels, cross-microphone synchronization,
or clock offset from generated transcripts alone. ASR can hallucinate, including
on silence; compare all null controls whether or not they favor the hypothesis.
No transcript is used to train a head or create labels.

## Provenance, resources, reload and reporting

New isolated output only. CPU600s/3GiB peak/64MiB outputs/1GiB disk reserve;
legacy8GiB GPU guard remains unchanged. Bound checks during generation, after
readback and full terminal source/product rehash; unknown monitor is FAIL/UNKNOWN.
Capture script/helper/test/protocol/input/model/library hashes before execution.
The library cohort includes the actual Whisper generate implementation, generic
GenerationMixin, GenerationConfig, logits processors and stopping criteria,
not only modeling_whisper.py; retain their captured hashes terminally.
Save config at creation; canonical row/array hashes; full frozen model state and
CPU RNG before/after; no gradients, trainable parameters or CUDA initialization.

First commit/test the runner. Then execute one diagnostic and a separate process
with a new output root, fresh decoder/model reload and the exact first result
SHA supplied explicitly. Recompute every PCM/mel/mask/token array and transcript,
retain the same source/model/profile and compare all outputs byte-exact.
This reuses the same implementation; do not call it independent ASR validation.

Report PASS/FAIL for execution/reload separately from cue interpretation. Preserve
failures; don't change constraints after observing outputs or overwrite a run.
Record full resident counts/bytes, environment, times/memory, all text/control
outputs and what remains UNKNOWN. Do not claim four-modality95–98%, decide a
product architecture from this pilot, quantize, resume legacy GPU or promote a
Teacher. Useful episode/task/split admission and true joint quality remain open.
