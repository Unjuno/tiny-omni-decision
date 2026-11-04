# Fixed audio dependence comparison and executed parity check

The question-only arm is already complete. Prepare audio-only and audio+question
arms with the identical scorer770→64→1 (49,409 parameters), seed17,24epochs,
AdamW .001/weight decay .001, batch128, same query order/sampler generator,
earliest minimum raw validation NLL, calibration-only17-point geometric
temperature grid .25..4. Each arm uses all2,312 training questions from578 sources
(55,488 presentations); validation560 questions from140 sources. All frozen
split/question bytes remain unchanged. Features are standardized from training
questions only, including query-weighted audio means/std. No sampling search.

For audio-only, question coordinates are zero. For joint, audio and question
each have384 coordinates. Encoders remain frozen; final24 and selected best
are saved separately. Preserve validation IDs/probabilities for all conditions,
compare Accuracy/NLL/Brier/ECE and grouped uncertainty by uploader. A joint gain
over question-only alone is insufficient if it does not also beat audio-only.
This is development model selection and cannot establish sealed generalization.

## Start gates

Complete media coverage and each file hash are mandatory. Decode audit must
match the completed media report; every waveform cache hash is checked.
Cross-split whole-template correlation screen covers exactly180,004 pairs:
578×138 +578×140 +138×140. Use mono2kHz decoded audio, shorter full waveform
as template against the longer, absolute normalized correlation over all
valid offsets. Threshold .99 or undefined correlation requires review.
Run4 CPU workers in bounded512-pair submission batches; record all scores and
offsets locally, pair-file hash, aggregate counts and maximum. This screen is
not proof against edits, shorter excerpts, time stretching or mixing.

Only completed matching screen with zero flags permits fitting. Audio features
must match exact source IDs/order and acquisition hash; all856 source features
required. This gate protects against accidentally treating a smoke as full data.

## Executed software checks

Ran the generalized scorer with all-zero audio using the original frozen
question data. Bestepoch8, temperature1.414214 and all560 validation probabilities
were bitwise identical to the original control; even serialized best weightSHA
was identical. No new configuration was selected. This parity run is explicitly
under `C:/CodexArtifacts/pretrained-reuse-r2/clotho-comparison-parity-01/` and is
not an audio/joint result. Added CLI was actually invoked on incomplete media;
it failed closed before creating an experiment directory or learning.
Ruff passed for the new scanner and comparison runner.

Commands after full acquisition:

1. `python -m scripts.audit_clotho_decoded_media`
2. `python -m scripts.check_clotho_cross_split_waveforms`
3. `python -m scripts.probe_clotho_audio_features --full-corpus --media C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-development-media-01 --output C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-audio-features-01`
4. `python -m scripts.train_clotho_audio_comparison`

Feature extraction uses the existing process-local Python dependency path.
No full-media screen, full features, audio-only or real joint run has executed
yet. Media acquisition remains active in the existing process. Next safe action
is to observe that process and perform these steps only once it terminates PASS.
