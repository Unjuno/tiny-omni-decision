# CLAP actual load and reusable audio branch

After the fixed Clotho-AQA readout comparison overfit, audit released contrastive
audio/text representations before another training run. No existing model,
corpus, selected checkpoint or sealed audit was changed/opened.

## Verified source and rights

LAION Hub model `laion/clap-htsat-unfused` pinned to
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`; model card declaresApache-2.0.
Official CLAP implementation snapshot
`1fd4c37df5ffbfcfbad5415c170bc66cf94c9994` hasCC0 code terms, a distinct scope.
Preserve both declarations/notices; do not infer weight rights from code alone.
No LAION training media was downloaded. Precise pretraining-record overlap is
unverified. Upstream describes checkpoint selection using Clotho/AudioCaps
retrieval scores; the HF checkpoint's exact original-file ancestry is unverified.
Future Clotho-AQA results therefore cannot be presented as blind to this influence.

Used installed HF CLI1.29.0 with fixed revision, no tool/environment update.
All10 repository files passed `hf cache verify --fail-on-missing-files`; its13
extra files were inspected and are downloader metadata under`.cache/huggingface`.
The first CLI invocation mixed filenames with include patterns, so JSON files
were separately requested before loading. No missing config was guessed.
Weight614,525,833bytes matches Hub LFS SHA:
`1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb`.

## Actual runtime and parameter inventory

Transformers5.6.2, Torch2.6.0+cu124, Python3.11.9, CPU FP32,4threads.
Local-only ClapModel load has empty missing/unexpected/mismatched/error lists.

| Loaded component | Parameters |
| --- | ---: |
| Text encoder | 124645632 |
| Text projection | 656384 |
| Audio encoder | 27534488 |
| Audio projection | 656384 |
| Parent scales | 2 |
| Full pair | **153492890** |
| Reusable audio+projection | **28190872** |

The full pair plus current114,365,184-parameter VideoPrism already totals
267,858,074 before other components, so it is a research reference rather than
a qualifying whole Omni model. The audio branch is a feasible compact reuse
candidate; complete resident inventory and quality still need measurement.

## Executed input and export checks

Used the first8s of licensed **training** source155281, rechecked WAV SHA,
native mono PyAV decode then torchaudio resample to48kHz. No question labels or
Decision metrics. This input is intentionally short for runtime testing.
Processor shape[1,1,1001,64]. Finite audio[1,512], text[3,512]; repeated audio
inference bitwise equal. Audio repeats plus three generic text prompts .311s
CPU excluding model load. No probability calibration or zero-shot accuracy claim.

Exported only audio_model/audio_projection tensors, loaded strictly into
ClapAudioModelWithProjection, then applied `F.normalize(audio_embeds,dim=-1)`
to match the full model's get_audio_features. Both full-model and independent
earlier-smoke comparisons were bitwise identical (max difference0). This native
audio class does not perform that final normalization itself. Export113,223,978
bytes; SHA `d190141ffd119b9096d9f3e096b2285f8ce6fa401cdb666d6f53911a2517fee3`.
Paired full/export inference .242s CPU excluding load. Model card and notices
are retained alongside export; no quantization or fine-tuning.

Initial smoke inference passed its assertions but report serialization failed
because loading_info contains sets. Its output and exact script are retained.
Converted sets to lists and reran in a new directory; this is a software repair,
not a new seed/configuration or a scored experiment.

## Open gate before a Decision comparison

The processor's maximum is10s and its default long-input policy is random
truncation. Current AQA clips are15–30s. Do not silently substitute a random10s
crop for whole-clip questions. First establish deterministic full-clip coverage,
record every window and aggregation, and compare with identical frozen IDs,
labels, budget and validation selector. Audio/text alignment, extra parameters,
different frontend and checkpoint-selection exposure are separate confounds.
No actual full-clip extraction or new Decision training has started here.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/clap-audit-01/` contains pinned
Hub/official metadata, weights/tokenizer/processor, both smoke directories,
strict audio export, exact executed scripts and hashes. Compact report and R1
inventory are committed; weights remain outside Git. This closes a load/reuse
audit, not the95–98% target, video/sensor gates or product release.

Sources: [pinned model card](https://huggingface.co/laion/clap-htsat-unfused/blob/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/README.md),
[pinned upstream README](https://github.com/LAION-AI/CLAP/blob/1fd4c37df5ffbfcfbad5415c170bc66cf94c9994/README.md),
[upstream code license](https://github.com/LAION-AI/CLAP/blob/1fd4c37df5ffbfcfbad5415c170bc66cf94c9994/LICENSE).
