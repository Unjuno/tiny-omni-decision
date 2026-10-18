# R1 local checkpoint identity recheck — 2026-10-18

This is a read-only SHA-256 and byte-count recheck of local files referenced by
the pinned R1 candidate inventory and the existing Path A package. It confirms
artifact identity only. It is not a new model load, forward pass, license
decision, runtime benchmark, or task-quality result. No checkpoint was changed.

| Role | Local artifact | Bytes | SHA-256 | Inventory match |
|---|---|---:|---|---|
| Video research reference (rights remain REVIEW) | `C:\CodexArtifacts\pretrained-reuse-r1\vjepa2_1_vitb_dist_vitG_384.pt` | 1,664,223,428 | `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` | exact |
| Path A image/video encoder | `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\components\image\videoprism-base-f16r288.npz` | 457,475,296 | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` | exact |
| Path A English text encoder | `C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4\components\audio\models\text-minilm-en\model.safetensors` | 69,569,488 | `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` | exact |
| Speech encoder-only export | `C:\CodexArtifacts\pretrained-reuse-r1\whisper-encoder-runtime-01\encoder.safetensors` | 32,839,984 | `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8` | exact |
| Japanese text candidate | `C:\CodexArtifacts\pretrained-reuse-r1\ruri-v3-30m\model.safetensors` | 146,828,152 | `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832` | exact |
| Environmental-audio candidate | `C:\CodexArtifacts\pretrained-reuse-r1\ast-audioset\model.safetensors` | 346,404,948 | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` | exact |
| Environmental-audio reference | `C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\dymn04_as.pt` | 8,086,973 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` | exact |
| Optional sensor encoder | `C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01\model\model.safetensors` | 151,615,328 | `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877` | exact |

The pinned MiniLM and VideoPrism files are the copies in the existing v4
assembly; the original assembly was not edited. Whisper's entry is the separate
encoder-only export, not the full encoder-decoder checkpoint. Ruri and MOMENT
remain task-quality-limited candidates; MOMENT's possible UCI HAR pretraining
overlap remains unresolved. V-JEPA 2.1 remains research-only because the exact
checkpoint grant is still under REVIEW. These rechecks do not change those
dispositions.

Reproduction used Python 3.11 standard-library `hashlib.sha256`, reading each
file in 8 MiB blocks, and compared both byte count and digest to
`docs/pretrained_reuse/R1_CANDIDATE_INVENTORY.json` or the v4 package manifest.
All eight files existed at the recorded paths; all eight byte counts and hashes
matched. No files were downloaded, opened as model weights, or rewritten.

GPU status was not required for this audit. The separate Orbitune workload was
left untouched. Any later GPU operation will use the existing HotUGPU handoff
gate: create its CUDA guard before waiting, retain the VRAM reservation across
child startup, and refuse launch unless its idle, process, utilization and free-
memory checks pass.
