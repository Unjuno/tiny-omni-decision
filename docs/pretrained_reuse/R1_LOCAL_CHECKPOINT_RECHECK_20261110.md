# R1 local checkpoint recheck — 2026-11-10

Read-only verification of the pinned VideoPrism and EfficientAT assets already
used by the Path A experiments. No model was loaded, no media or feature cache
was opened, and no GPU work was run during this recheck.

## Checkpoint and source integrity

| Component | Pinned source | Local asset | Bytes | SHA-256 | Result |
|---|---|---|---:|---|---|
| VideoPrism-B f16/r288 | Hub `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`; code `google-deepmind/videoprism@c6eba0ece3dd49a24ffc44582966997a4a59f1d8` | `C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-f16r288.npz` | 457,475,296 | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | Exact byte length and hash match |
| EfficientAT DyMN-0.4 AudioSet | Code `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806` | `C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt` | 8,086,973 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | Exact byte length and hash match |

The local source checkouts report the exact pinned commits. VideoPrism's source
checkout has pre-existing untracked build, egg-info, and Python cache outputs;
they were left untouched. The EfficientAT source checkout is clean. The local
VideoPrism license file is Apache-2.0, and the local EfficientAT license file
is MIT. The pinned Hub API reports `cardData.license=apache-2.0` for the exact
VideoPrism revision. EfficientAT's maintainer explicitly confirms that the
pretrained checkpoints may be used commercially in [upstream issue #44](https://github.com/fschmid56/EfficientAT/issues/44).

## Runtime and quality interpretation

This was an integrity and rights recheck only; it does not claim a new runtime
measurement. The prior R1 record documents finite local inference for both
encoders and their bounded runtime measurements. Existing task results remain
task-specific: VideoPrism is not evidence of broad video Decision quality, and
EfficientAT's prior STARSS23 result did not show a benefit from adding video.
The four-component Path A package was previously measured at about 687.3 MB,
above the provisional 200 MB target. No compression or product promotion is
justified by this recheck.

Primary sources: [pinned VideoPrism model revision](https://huggingface.co/google/videoprism-base-f16r288/tree/be719a406d563b66f0ac969e7c94bab8e997c81a), [pinned VideoPrism source license](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/LICENSE), [pinned EfficientAT source license](https://github.com/fschmid56/EfficientAT/blob/a425fdce92572e602a1d5634799bd9f1f2efa806/LICENSE), and [checkpoint-use confirmation](https://github.com/fschmid56/EfficientAT/issues/44).
