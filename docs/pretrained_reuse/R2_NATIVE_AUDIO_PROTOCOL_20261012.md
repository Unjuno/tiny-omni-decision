# Frozen native speech readout

Reuse the existing pinned Whisper Tiny generation decoder instead of discarding
it and classifying pooled audio features. This tests a task-matched pretrained
readout, not general spoken reasoning. No added model or training. Full decoder
parameters must be counted, with tied output embeddings counted once.

Freeze this before inference: same 1,938 LibriSpeech dev-clean clips, full text,
four options/order/targets in both original random and hard-negative conditions.
Input SHA1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923.
Whisper revision169d4a4341b33bc18d8881c4b69c2e104e1cc0af; weight
SHA7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395.

FP32, eval, no gradients. Mono16k PCM decoded identically from FLAC; pad30seconds,
fail if truncation needed. English/transcribe/no-timestamps prefix. Preserve
transcript case/text; teacher-force each candidate. Exclude prefix and padding
from log likelihood; include EOS. Primary score is mean transcript-token log
probability (including EOS); softmax across four options, temperature1. No choice
of normalization after seeing outcomes. Length normalization is a scoring heuristic,
not a calibrated posterior. Save scores, token counts and all probabilities.

Hash every audio and model/processor input. Separate eight-clip smoke verifies
finite scores, batching/option-permutation invariance and parameter count before
the full run. Same first eight clips remain in the full development diagnostic;
this is not blind. Smoke may only repair implementation, not tune scoring.
New output directories, all existing artifacts read-only.

Report Accuracy/NLL/Brier/ECE for both conditions and unchanged pooled-readout
baseline on exactly the same IDs. Paired speaker bootstrap,10,000replicates,seed17,
percentile95% Accuracy difference hard-minus-random;40speaker groups. Save full
metrics, input/runner/environment hashes, time/VRAM and unchanged weight hash.
No training, generation, prompts search, temperature calibration, sealed audit,
dataset expansion or product promotion. Audio transcript matching alone cannot
establish95% general audio decisions or environmental-sound understanding.
