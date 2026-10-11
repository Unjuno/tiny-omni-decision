# Native pretrained speech readout results

Execution PASS. On the same1,938 LibriSpeech dev-clean utterances and unchanged4-choice sets, native Whisper Tiny transcript likelihood gives1,937correct for both random and hard distractors. No weights were trained or changed. Sourcefded2d8 (matched/smoke),51840dd (post-hoc rotated-audio control).

| Readout / audio | Distractors | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Existing pooled readout | random_distractors | 0.265222 | 2.608463 | 1.088013 | 0.448658 |
| Existing pooled readout | hard_distractors | 0.263674 | 2.536068 | 1.084071 | 0.442511 |
| Native decoder / matched | random_distractors | 0.999484 | 0.108266 | 0.019962 | 0.099431 |
| Native decoder / matched | hard_distractors | 0.999484 | 0.107729 | 0.017809 | 0.100230 |
| Native decoder / rotated | random_distractors | 0.248194 | 1.503682 | 0.800583 | 0.151235 |
| Native decoder / rotated | hard_distractors | 0.220330 | 1.497658 | 0.803624 | 0.155975 |

## What this establishes

A task-matched pretrained decoder recovers useful acoustic/transcript alignment lost by the existing pooled-feature readout. Fixed cross-speaker half-corpus rotation reduces accuracy to near chance; all1,938 original assets remain present, and the choices/targets stay unchanged. This supports acoustic dependence, not general audio reasoning. Previous low transfer did not justify training a new encoder or increasing parameters.

Both distractor conditions saturate on this transcript task. The current phoneme/lexical/length heuristic is harder by construction, but empirically not a useful challenge to native Whisper here. Do not claim arbitrary hard-negative robustness or change this dataset to rescue that claim. Separate future task design is needed for spoken conditions, environmental events, ambiguous minimal pairs and OOD/noisy speech.

Mean-token log likelihood (including EOS; prefix/padding excluded) is softmaxed at temperature1. Confidence is around90% despite99.95% accuracy; ECE~0.10 remains an underconfidence issue. No temperature was fitted on these development records. These normalized scores are not proven calibrated decision probabilities.

Paired hard-minus-random accuracy bootstrap uses40speakers,10,000replicates,seed17: difference0, interval[0,0], because exactly the same single example is wrong in both conditions. This interval describes this paired difference only, not certainty in99.95% population accuracy. Public pretraining overlap is UNKNOWN; evaluation is observed development data, not sealed audit.

The rotated-run JSON also emits the generic query-speaker bootstrap; source audio now couples different speaker groups, so that interval is not used for inference. Its metrics are a descriptive dependence control.

## Resources and reproducibility

Model37,760,640resident parameters, with tied decoder output/embedding counted once. This full model was already in the parameter tally of the old encoder-only runtime; using its decoder adds no resident model parameters. Text MiniLM/readout are not used for this transcript scoring. This diagnostic is not yet a generic multimodal runtime replacement.

Matched run 63.737s including model load/preprocessing and scoring; rotated 64.124s; peak CUDA allocation507,187,200bytes. FP32, RTX3080Laptop16GB, no cloud cost. End-to-end amortized matched time32.89ms/utterance for both candidate sets; this is not isolated single-request latency.

The initial invocation failed before output creation because SoundFile was absent. Installed only into a new isolated directory: soundfile0.14.0,cffi2.0.0,pycparser2.23; existing environment unchanged. Runtime command uses PYTHONPATH=C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;src. Exact args, versions, UTC times, processor and weight hashes are in companion JSON.

Eight-clip smoke passed option-permutation and single-versus-batched parity (maximum absolute difference<2e-6). All eight are still included in the main diagnostic, not treated as independent blind samples. CPU tests359passed1skipped; Ruff pass. Unit tests cover shifted target mask, EOS inclusion, prefix/pad exclusion, lengths/nonfinite guards and deterministic speaker rotation.

Local outputs: C:/CodexArtifacts/pretrained-reuse-r2/native-speech-smoke-01, native-speech-eval-01, native-speech-rotated-01. All probabilities and option token counts retained locally; SHA256s tracked in companion JSON. Existing baseline predictions, model weights, media and training artifacts unchanged. No sealed audit read, no product promotion, no compression.

## Next decision

Retain native speech likelihood as a candidate transcript-decision route. Before broad integration, separately test whether a question-conditioned readout can answer spoken content and conditions; matching an offered verbatim transcript is not sufficient. Preserve general text/image/video and all-modality95–98% goals as unmet. Quantization remains downstream of usable task quality.
