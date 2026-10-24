# Frozen native coverage run dependency snapshot

These five raw source files are bundled because the completed repeat-fit
receipt reports source commit `b3b7bb3abc4d27b6da51aa05032d2ff14704362f`, but
their bytes do not match the corresponding files in that Git tree. Their bytes
do match the run's `validated_hashes` entries. The exact historical Git commit
that contained this complete dependency set is unknown; this snapshot preserves
the validated runtime bytes without attributing them to an unsupported commit.

| File | SHA-256 |
| --- | --- |
| `scripts/native_fit.py` | `62315a27e37c9b3c2837ee66c2d462c4285b4aca1a739fb769974c3330f8951b` |
| `scripts/native_fit_runtime.py` | `63598df78a632b6a2a49c4d70f9c9e07a945f94793e07cb4144ec2234f346abe` |
| `scripts/native_candidate_inputs.py` | `372e8c3664fbfa8627a9bba4ff4c7ab3c0f3fc92d8a0c5df3e8011b1a45cd7f3` |
| `scripts/native_video_cache.py` | `62ffe34c6db4a52ed93310c36939a74f356729db338a7aea49406f1cdfe58039` |
| `tiny_omni_decision/native_candidate.py` | `cd2ab4e45de5c5e3e037531bf8a9e967365f8ef2a60b5b20cddd0ba3e336a974` |

The diagnostic loads the model, feature-store, and position helper modules from
these files by explicit path after verifying each digest against the run
receipt. It reimplements the sampler's short deterministic `option_order`
function; that function is preserved verbatim in behavior from the validated
`native_fit.py`. The snapshot is diagnostic provenance, not a replacement for
the current project modules.
