# Speech Commands multi-query audio-feature dependence diagnostic

## Scope

This is a post-hoc, validation-only ablation of the saved frozen-feature
Audio+Text scorer. It compares the original Whisper feature for each clip with
(1) a zeroed audio vector and (2) the Whisper feature from a different
validation clip. Text embeddings, question and option wording, targets, readout
weights, option order, and validation IDs stay fixed. All four query types for
one clip receive the same audio intervention.

No training or optimizer update was run. The sealed audit and test split were
not loaded. This experiment measures whether this fixed readout depends on its
cached audio representation; it is not model selection, blind evaluation,
general speech QA, or synchronized audio-video evidence.

## Frozen inputs and integrity

- Source experiment: `path-a-audio-multiquery-speech-commands-v0`, selected
  epoch 4, one shared `CandidateScorer` checkpoint; source run report SHA-256
  `833258cbc584cbe5f2583c7cc9a5a6f115deb9a46b232e81b3eb7c7489fc081b`.
- Config SHA-256:
  `d5b130b8ba859a21091cd2fd6ff327301d408ae1cd0a4ae83e9a282708843396`.
- Readout checkpoint SHA-256:
  `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.
- Validation manifest SHA-256:
  `18520e5d422fa4a129aa1c9ef3295acd4f00c97c0ca3493cb972c28aafa7d0bb`;
  ordered validation ID SHA-256:
  `20dca814ebf78e19ce0d81d77a2f87fec615d9496c20b7567347347ec9073b70`.
- Whisper Tiny revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, weights
  SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- MiniLM-L3 revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, weights
  SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- The v4 immutable observation-cache report and manifest hashes are
  `459871c66419122fcae96e3ed542d4e87dc7d0a30b73bfc850bcef998ec79f5a` and
  `fe0d8b5717651d7e845e577902c6230f2f96516b654870310433bb2c648d3f39`.
- Before intervention, reloaded baseline predictions matched the saved
  prediction identities and order. Maximum logit delta was `1.15e-5`; maximum
  probability delta was `1.94e-6`. The audio features also matched the
  question-independent persistent cache and the original frozen feature file
  bit-for-bit.
- Train and validation have zero speaker and media identity overlap. The
  validation set has 640 clips, 640 unique audio assets, 206 speaker groups,
  and 2,560 queries (four per clip). All 640 validation WAV files were
  re-hashed before the run.

The successful corrected run used seed 17 on CPU, Python 3.11.9, PyTorch
2.6.0+cu124, and Transformers 5.19.0. The diagnostic took 145.3 seconds,
including the 10,000-replicate bootstrap. No GPU or cloud compute was used.

The first diagnostic attempt is retained at
`C:/CodexArtifacts/pretrained-reuse-r2/audio-multiquery-audio-dependence-v0/`
with an `INVALIDATED.txt` marker: its random permutation had two fixed points.
Its shuffled-audio result must not be cited. The corrected run below uses a
true derangement and a separate output directory.

## Results

All metrics are unweighted means over the four question types; each type has
640 queries. Brier is the sum of squared multiclass probability errors.

| Condition | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE-15 |
| --- | ---: | ---: | ---: | ---: |
| Original audio | 0.9785 | 0.0693 | 0.0323 | 0.0143 |
| Audio zeroed | 0.5250 | 1.2478 | 0.5713 | 0.1769 |
| Other-clip audio | 0.4961 | 4.0821 | 0.9709 | 0.4835 |

| Question type | Condition | Accuracy | NLL | Brier | ECE-15 |
| --- | --- | ---: | ---: | ---: | ---: |
| Keyword identity | Original | 0.9594 | 0.1493 | 0.0610 | 0.0248 |
| Keyword identity | Zeroed | 0.1000 | 2.6535 | 0.9656 | 0.1620 |
| Keyword identity | Other clip | 0.1047 | 9.4081 | 1.7143 | 0.8484 |
| Direction presence | Original | 0.9797 | 0.0596 | 0.0319 | 0.0114 |
| Direction presence | Zeroed | 0.4000 | 0.7493 | 0.5553 | 0.1941 |
| Direction presence | Other clip | 0.5172 | 3.0238 | 0.9300 | 0.4642 |
| Response presence | Original | 0.9812 | 0.0422 | 0.0225 | 0.0093 |
| Response presence | Zeroed | 0.8000 | 0.6912 | 0.3734 | 0.1634 |
| Response presence | Other clip | 0.6938 | 1.8254 | 0.6019 | 0.3019 |
| Switch presence | Original | 0.9937 | 0.0262 | 0.0139 | 0.0119 |
| Switch presence | Zeroed | 0.8000 | 0.8971 | 0.3908 | 0.1882 |
| Switch presence | Other clip | 0.6687 | 2.0711 | 0.6372 | 0.3194 |

The paired speaker-cluster bootstrap sampled the 206 validation speaker groups
with replacement, keeping each sampled speaker's clips and all four queries
together. It used 10,000 replicates and seed 17. Intervals below are percentile
95% intervals for *condition minus original audio*; negative NLL/Brier means
the intervention is better.

| Intervention | Metric delta | Mean paired difference | 95% interval |
| --- | --- | ---: | ---: |
| Audio zeroed | Accuracy | -0.4535 | [-0.4680, -0.4386] |
| Audio zeroed | NLL | +1.1785 | [+1.1388, +1.2171] |
| Audio zeroed | Brier | +0.5390 | [+0.5207, +0.5566] |
| Audio zeroed | ECE-15 | +0.1600 | [+0.1460, +0.1737] |
| Other-clip audio | Accuracy | -0.4823 | [-0.5020, -0.4623] |
| Other-clip audio | NLL | +4.0122 | [+3.8352, +4.1923] |
| Other-clip audio | Brier | +0.9382 | [+0.9002, +0.9760] |
| Other-clip audio | ECE-15 | +0.4674 | [+0.4482, +0.4866] |

The corrected seed-17 shuffle is a full derangement: all 640 clips receive an
audio vector from another clip, with zero self-assignments. Five source clips
share a speaker group with the receiving clip; all are still different clip
IDs. The permutation SHA-256 is
`a0b3b435b07a309dae209a92b396051b2f965e4539517c5a70e7e5fbae231e6e`.

## Interpretation and limits

For this frozen readout and narrow Speech Commands task, audio features carry
substantial predictive signal: zeroing them removes most keyword identity
performance, and replacing them with another clip's vector damages every
macro metric. The response/switch tasks retain 80% accuracy with zero audio,
which reflects label frequencies and task structure; it is not evidence that
the text alone recognized those clips. The extreme NLL under shuffled audio is
an out-of-distribution feature/label mismatch and should not be interpreted as
a natural chance-level score.

The same development validation was used to select the checkpoint, so these
post-hoc intervals do not make the data an independent test. The experiment
supports feature dependence only for the fixed checkpoint, split, and four
closed-set queries. It does not establish broad speech understanding,
synchronized audio-video fusion, or the final product quality gate. A separate
task-matched, previously unused evaluation is needed before making those
claims.

The machine-readable report and per-condition predictions are outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/audio-multiquery-audio-dependence-v1/`.
Report SHA-256:
`b4bfc480df9a2e018249b2721d0b4796e4c46e3112df6524b80858c5facca785`.
The exact runner is `scripts/diagnose_audio_multiquery_audio_dependence.py`;
runner SHA-256 `fbd5de3b40eaae67e0ad17993fab9f27436c7e6636b386faa5d2cae6265f9541`.
The invocation and output hashes are captured in the external
`reproduction-metadata.json` (SHA-256
`cc7a9edcd3a033cd506e47d92793b235ff58d3bc0980f8fc095b30b153db9584`).
