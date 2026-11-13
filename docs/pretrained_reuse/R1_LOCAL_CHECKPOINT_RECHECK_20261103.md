# R1 local VideoPrism / EfficientAT checkpoint recheck — 2026-11-03

This is a read-only integrity and rights recheck for the two frozen encoders in
the RoboMNIST AV-match Path A. No model was loaded or executed during this
recheck; the earlier CPU runtime smoke remains the runtime evidence recorded in
[`R2_ROBOMNIST_AV_MATCH_PREFLIGHT_20261103.md`](R2_ROBOMNIST_AV_MATCH_PREFLIGHT_20261103.md).

| Encoder | Pinned revision | Local bytes | Expected and actual SHA-256 | Rights evidence |
|---|---|---:|---|---|
| Google VideoPrism-B f16/r288 | Hub revision `be719a406d563b66f0ac969e7c94bab8e997c81a`; source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` | 457,475,296 | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | Exact Hub revision API returns `cardData.license=apache-2.0`; pinned official source LICENSE is Apache-2.0. |
| EfficientAT DyMN-0.4 AudioSet | Source commit `a425fdce92572e602a1d5634799bd9f1f2efa806` | 8,086,973 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | Pinned repository LICENSE is MIT. The maintainer's checkpoint-specific response in issue #44 explicitly permits commercial applications. |

Primary rights sources: [pinned VideoPrism model card](https://huggingface.co/google/videoprism-base-f16r288/tree/be719a406d563b66f0ac969e7c94bab8e997c81a), [pinned VideoPrism source license](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/LICENSE), [pinned EfficientAT source license](https://github.com/fschmid56/EfficientAT/blob/a425fdce92572e602a1d5634799bd9f1f2efa806/LICENSE), and [maintainer checkpoint permission](https://github.com/fschmid56/EfficientAT/issues/44).

Both byte lengths and hashes match
[`R1_CANDIDATE_INVENTORY.json`](R1_CANDIDATE_INVENTORY.json). VideoPrism's
local source checkout is at the pinned commit but has only untracked generated
`build/`, `videoprism.egg-info/`, and `videoprism/__pycache__/` paths; those
were not changed. EfficientAT's local source checkout is clean at its pinned
commit.

## Cost interpretation

The two encoder checkpoints total **465,562,269 bytes** (about 444 MiB) and
**116,326,423 frozen parameters**. The existing bilinear AV readout contract
has 73,728 trainable parameters, making the current Path A architecture
116,400,151 resident parameters before counting serialization metadata. The
raw checkpoint total is above the provisional 200 MB package target; no
quantized export or package-size claim is made here. Quality evaluation,
save/reload of a trained readout, and the 1,780-group validation remain open.

No weights, source checkouts, media, feature cache, or run state were modified.
This recheck does not admit any dataset by itself and does not resolve the
current C: capacity gate.
