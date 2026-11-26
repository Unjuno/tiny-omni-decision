# Fixed acoustic collection-cue diagnostic — executed, uninformative

**Execution/reload PASS; cue interpretation UNKNOWN.** The three real recordings
and all three same-length silence controls produce the identical text ` you`
and complete token IDs. This establishes neither speech nor counter absence.
Do not adjust gain/prompts/models/settings on this pilot to manufacture an answer.
Joint Decision quality and the full Omni95–98% goal remain unmeasured here.

## Fixed experiment and actual outputs

[Prospective protocol](R2_ROBOMNIST_AUDIO_CUE_PROTOCOL_20261015.md), source commit
`e1d1cb57f55b02c849d76baccc3928011b3186f4`; known TRAIN recording
`Ar1_Vhigh_Ac0_Rp01_UNC062`, all three existing complete WAV files. No additional
media, position GT, benchmark selection or sealed audit. Source bytes and native
decoded PCM match the prior immutable input-bank/replay.

| Receiver | Native44.1kHz frames | Resampled16kHz samples | Real text | Silence text | Max real/null mel difference |
|---|---:|---:|---|---|---:|
|1|658432|238887|` you`|` you`|1.911504|
|2|659456|239259|` you`|` you`|1.842933|
|3|660480|239630|` you`|` you`|1.926805|

PyAV18.1.0/libswresample6.3.102 converts the complete mono float32 decoded PCM,
flushes its output and preserves duration within one16kHz sample; no clipping,
gain normalization, cropping, counter removal or synthesized common clock.
Native Whisper mel80×3000 tensors and attention masks are saved. Real input
tensors **do** differ from zeros, despite the same decoded words. These results
do not establish that acoustic encodings lack useful robot information, a general
Whisper capability ceiling, or absence of non-speech sounds.

Offline full `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`:
weights SHA `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
All11 model/tokenizer/processor files match the pinned historical reference.
Count **37,760,640** resident parameters /151,042,560 parameter bytes, including
decoder. Zero trainable; CPUfloat32/eager/2threads/seed17, greedy English onebeam,
temperature0.0, max_new_tokens128, no timestamps/fallback threshold search.
Full returned IDs retain the native prefix and terminal EOS; each condition uses
two decoder forwards (12 total). The native generation configuration is copied;
the base config remains unchanged. Full parameter/buffer state before/after:
`e133c077072cfa2636a82db2fb1fae31e26cb4fd7d40d497db77b0667609224a`.
Forward CPU RNG unchanged, no gradients or CUDA initialization.

The frozen Hub weights' Apache-2.0 and original OpenAI MIT source terms remain
distinct; the [RoboMNIST source](https://doi.org/10.6084/m9.figshare.28179383.v1)
uses CC BY4.0. Upstream collection code was inspected, never executed or adopted.

## Reproduction and saved evidence

```powershell
python -m scripts.probe_robomnist_audio_cue --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-audio-cue-01
python -m scripts.probe_robomnist_audio_cue --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-audio-cue-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/robomnist-audio-cue-01 --reference-sha256 884feba63bf1e203a588356c53b5ed3e12e5d26d8a846b90dbb490724f274cc4
```

| Execution | Seconds | Peak CPU working set | Bound evidence |
|---|---:|---:|---|
| Main |8.269711|934,744,064 bytes|133 pins /21 arrays |
| Fresh separate-process replay |8.641818|933,564,416 bytes|137 pins /21 arrays byte-exact |

The replay reopens native audio and loads a fresh model, reproducing all PCM,
mel, mask and token arrays, transcript/token/accounting fields except elapsed
generation times, plus full state and environment exactly. Whole transcript JSON
hashes differ because timing is recorded separately for each execution.
It shares the ASR implementation and does **not** independently validate speech.
No model or Teacher was trained, GPU restarted or paid/cloud compute used.
Electricity cost was not measured. Source/profile/results remain immutable.

Main result SHA `884feba63bf1e203a588356c53b5ed3e12e5d26d8a846b90dbb490724f274cc4`;
config `52f1c712cd94f4ad5c19809ae1d1b5e4d973c1e44c13dc2126d472209e124e2d`;
arrays `63c71c32a318ec75d659f7f29a1f05803bc0d4f601364e2e766ab4b326ec85b1`.
Replay result SHA `2a0efaf08b486aed0326365bf8d3853ea670b78aa90fc9c0f0b3c7aac7455135`.
[Small exact summary](R2_ROBOMNIST_AUDIO_CUE_RESULTS_20261015.json) retains every
transcript/token row, artifact hashes and measurements. Media/weights/arrays are
outside Git at the command paths. This prototype adds no deployed Omni parameters.

Preexecution inspection corrected the installed short-form tensor API's prefix
removal by requesting complete ModelOutput sequences; no prefix was reconstructed.
Review also found the actual Whisper/generic generation implementations separate
from the model class file. Decorated and unwrapped generation, configuration,
logits and stopping modules are now pinned before load/decoding, with terminal
and fresh rechecks. Regression tests cover both issues.

Observed library warnings remain recorded: mixing generation_config and explicit
kwargs is deprecated; max_new_tokens128 overrides native max_length448; native
suppression processors take precedence over duplicate processors. The effective
base config and explicit kwargs are both saved. No dependency/settings change
was made to hide warnings or improve the observed transcript.

## Next decision

This ASR probe provides no evidence to settle the collection-counter hypothesis.
Retain `UNKNOWN` for its presence and `UNVERIFIED` for common capture clock and
speech truth. Do not admit finely synchronized RoboMNIST fusion from these words.
An episode-level operation task still needs explicit semantics, source/content
grouping and informative unimodal/joint controls; no Decision metrics are claimed.
The public-data [VisTouch source screen](R2_VISTOUCH_SOURCE_RESULTS_20261015.md)
examines a distinct material/contact use case with its own timing/data limitations.
