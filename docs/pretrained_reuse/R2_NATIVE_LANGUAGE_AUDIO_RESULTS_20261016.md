# Native language/audio input execution: smoke success, full cache incomplete

2026-10-16. Actual code and model execution following the strategy review.
Language input processing and the common speech/event frontend are implemented.
**The full language cache is INCOMPLETE due to disk capacity. The shared Decision
readout remains untrained.** The95–98% and full native/joint/OOD goals remain open.

[Exact commands, times, resource counts and receipt hashes](R2_NATIVE_LANGUAGE_AUDIO_RESULTS_20261016.json)
· [Language protocol](R2_NATIVE_LANGUAGE_CACHE_PROTOCOL_20261016.md)
· [Audio protocol](R2_NATIVE_AUDIO_PROFILE_PROTOCOL_20261016.md).

## Language: measured success and limit

`native_language.py` preserves the original untruncated token sequence. Short
observations use the original forward. Long observations use510content-token
windows with32overlap, retaining each output token once, one initial CLS and
one final SEP. This preserves token coverage, not full-context attention or
proven cross-window reasoning. Query/candidate pairs remain original paired
inputs with a512-token rejection gate and masked mean. No text is rewritten.

The existing six development files contain29,164questions,4,382unique raw states
and22,328unique question/candidate pairs. All rows and option order map through
the new cache index; train/validation state banks are separate. Normalized-state
and group overlap are rejected. Thirty-eight states exceed512tokens. These are
inventory counts, **not completed full feature extraction**.

The actual smoke encoded4TRAIN states, including the753-token longest TRAIN
observation, and8TRAIN query/candidate pairs. Historical short-observation output
and the original single-pair pooling convention both match exactly, maximum
absolute difference0. The four states/eight queries pass file save/reload.
The original70,830,722-parameter language model's full state hash is unchanged.

| Execution | Result | Process seconds | Peak CUDA allocation | OS peak working set |
|---|---|---:|---:|---:|
| Language smoke | PASS | 12.7134 | 371,848,704 bytes | 2,052,997,120 bytes |
| Full language cache attempt | INCOMPLETE before model load | 8.7090 | No model loaded | Not recorded |

Full planned tensors: train `[375078,384]`, validation `[237067,384]`, query
`[22328,384]`, all FP32: **974,550,528 bytes**, excluding file headers/index.
Each full index is56,302,070bytes. When the full run reached the disk gate,
available space could not hold all tensors plus the fixed2GiB reserve. A following
snapshot showed2,341,109,760bytes free. No state/query was encoded by that attempt,
and no full feature banks were allocated. Preserve the failed directory, index,
effective config, traceback and `result.json`; it is not a training input bank.

The source of the wider free-space decrease is UNKNOWN. Do not attribute it to
the run, pagefile or another application without evidence. No existing data,
checkpoints, test scratch or system files were deleted to force the run through.
The capacity threshold and feature precision were not changed.

## Audio: actual unified preprocessing and fresh replay

While capacity-dependent language extraction was blocked, implemented the common
audio primitive against the existing native integration. Both speech and event
features now derive from each admitted mono16k/24k waveform.

- Original16k waveform is preserved for speech. STARSS24k becomes16k with the
  installed CPU sinc resampler; official Whisper30s padding is retained, but
  output keeps only ceil(physical samples/320) tokens.
- Event audio starts directly from the original16k/24k waveform and resamples
  to32k on the model device. No24->16->32 conversion, amplitude normalization,
  transcript, label input or silent cropping.
- STARSS uses original FOA channel0 over the original5s window. This does not
  establish spatial audio understanding. Historical STARSS event caches were
  CPU-based; exact parity with those banks has not been established.

| Fixed TRAIN input | Physical samples | Speech tokens | Event features | Evidence |
|---|---:|---:|---:|---|
| Lombard `s11_l_bbab2n` | 35,200 at16k | 110x384 | 1x384 | Both features exactly match saved native integration |
| STARSS `fold3_room12_mix001:t0000-0005` | 120,000 at24k | 250x384 | 1x384 | New complete two-path output; no historical speech reference |

A fresh process recomputed both examples. All6PCM/feature arrays reproduce
exactly; a separate CPU readback also rehashed every saved artifact and checked
all6arrays for dtype, finiteness and equality. Original pinned source/input hashes
and frozen model states match before/after. Speech8,208,384parameters,
event1,961,239,mel0. No new learned weights or optimizer updates.

Audio smoke/replay took5.9232/6.1635seconds; peak CUDA allocation200,990,208bytes
both. OS peak working set1,394,356,224/1,395,695,616bytes. These are two-input
process measurements including setup/hash work, not corpus throughput or
end-to-end Decision latency. No quality labels or calibration/validation/audit
inputs were used for these model forwards.

## Environment and verification

Local RTX3080Laptop16GiB; Python3.11; Torch/torchaudio2.6.0+cu124, CUDA12.4,
Transformers5.6.2, NumPy2.4.6, safetensors0.8.0, soundfile0.14.0. FP32,
eager Whisper/DeBERTa,TF32off,batch1,deterministic algorithms. No package upgrades.
The tokenizer emits its standard over512 warning while inspecting full lengths;
no over512 encoder window is forwarded. Existing audio dependency deprecation
warnings are retained in logs and do not indicate a runtime upgrade.

- 42new CPU tests: language/window/cache31 and audio11. Missing implementation
  was observed failing before implementation; all pass afterward. Audio tests
  require the existing isolated torchaudio path, which is optional in normal CI.
- Full local suite with that existing runtime: **1,971passed,1skipped in32.05s**.
- Ruff passes. All25existing manifest validation/audit commands pass.
- Language source `c83ebfca7b0f79a8731cd51ff7740f7d657ec916`,
  [CI1807 success](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38853488163).
- Audio source `7130765d73d0cb4b80c4c6dc4066d585d5010fa2`,
  [CI1809 success](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38854731419).
  Final publication CI is checked separately after push.
- Local artifacts are under `C:/CodexArtifacts/pretrained-reuse-r2/`:
  `native-language-smoke-01`, `native-language-cache-01`,
  `native-audio-profile-smoke-01`, `native-audio-profile-replay-01`,
  `native-language-verification-01`, `native-language-audio-results-01`.

Total recorded four-process wall time33.5090seconds; excludes preparation,
tests and the independent readback. GPU-kernel time/electricity were not measured.
Cloud cost0. All processes exited; no background training was scheduled.

## Remaining work and concrete restart conditions

Full language extraction/replay, full paired audio features, common data/sampling/
selection contract and shared-readout training remain unexecuted. No model-quality
or accuracy improvement is claimed from these input primitives.

Capacity request sent: approximately6GB free on C: or another specified output
volume. Language alone needs roughly3.2GB free before its new index and arrays;
later audio needs additional storage. Current artifacts and resource gates stay
protected. Recheck actual free space and live processes before a new invocation.

After capacity exists, use fresh `native-language-smoke-02` and
`native-language-cache-02` directories. First rerun the language smoke on the
then-current clean committed source, then pass that successful receipt and its
SHA-256 to `capture_native_language --phase cache`. New Python files changed the
source inventory, so the old smoke must not be misrepresented as the same-source
reference. After full cache success, run `--phase replay` against its receipt/hash
in a separate process/output. Do not resume the failed allocation as completed.

Then use the now-exercised audio frontend to finish the two-source feature banks
with train/validation isolation and explicit source/sample accounting. Preserve
all original corpora, checkpoints and old numerical profiles. Freeze the common
fit contract before the first shared Decision update.
