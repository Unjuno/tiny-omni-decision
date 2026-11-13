# R1 release and license follow-up (2026-10-11)

This follow-up updates the pinned inventory using primary upstream sources and
one local weights-only audit. It does not train, tune, or inspect a sealed
evaluation set.

## Newly resolved: EfficientAT DyMN-0.4 AudioSet

- Code source: `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806`.
  The release asset is `dymn04_as.pt` from GitHub release `v0.0.1`, 8,086,973
  bytes, SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`.
- The upstream repository is MIT licensed. More specifically, the owner
  answered the checkpoint rights issue on 2026-09-07: “feel free to use the
  EfficientAT pretrained models and checkpoints for commercial applications.”
  This resolves the previous uncertainty about commercial use of the released
  checkpoint.
- Strict local load succeeded from the downloaded release asset using the
  pinned source tree. Measured model: 1,961,239 parameters, 7,844,956 FP32
  parameter bytes. A deterministic, non-silent 2-second waveform passed the
  official `AugmentMelSTFT` path and produced finite `[1, 527]` logits and
  `[1, 384]` pooled features. Local RTX 3080 Laptop forward took 0.366 s and
  peaked at 18,560,512 allocated CUDA bytes in this single diagnostic.
- This confirms technical and checkpoint-rights feasibility for environmental
  audio tagging. It does **not** establish Speech Commands keyword Decision
  quality, general audio understanding, or joint audio-video performance.
  Keep it as a small ALLOW audio candidate; select it only through the same
  task-matched Decision evaluation as other encoders.

## Release status across remaining R1 candidates

| Candidate | Release status | Rights/runtime finding | R1 disposition |
|---|---|---|---|
| EfficientAT DyMN-0.4 AudioSet | pretrained and accessible | MIT source plus explicit upstream commercial checkpoint permission; pinned release asset loaded locally | ALLOW research candidate; task quality open |
| ViT-Lens-L | pretrained and accessible | Upstream repo says it released models; pinned Hub files total 12.326 GB and are paired to original CLIP ViT-L/14. Not compatible with V-JEPA; no local load or matched Decision task | Research/reference only; too large for target |
| PE-Core T/16, S/16, B/16 | unknown (pretrained checkpoint pages are listed; asset access was not verified) | Official docs list all three checkpoint names and Apache-2.0 checkpoint terms. The anonymous Hub API audit returned HTTP 401; exact asset hashes/sizes and local loads are unknown. The repository's implementation package separately declares FAIR Noncommercial Research License. | Keep checkpoint terms and implementation terms separate; no product use of the official implementation without a license-clean route. T/S are compact research candidates if artifact access and the implementation path are resolved. |
| VATT | weights listed, but access not verified | Official README lists Google Storage checkpoints, but the link redirects to Google sign-in in this audit. The README does not state an exact weight license; old documented runtime is Python 3.8/CUDA 10.1/TensorFlow 2.7 | UNKNOWN pending anonymous access, exact weight terms, and runtime feasibility |
| MJEPA | code only | Official repository is MIT and documents large ViT-L/ViT-g training/evaluation. The official Releases page has no releases or checkpoint assets | Architecture reference only; no pretrained checkpoint to audit |

“Pretrained and accessible” means a primary-source weight artifact could be
retrieved or independently verified as accessible in this audit. Where a
model page lists weights but the file could not be accessed or pinned, the
inventory says “unknown” and records the listing separately. This does not
establish task quality, device fit, or product integration. Checkpoint and
source-code terms are recorded separately where they differ.

## Reproduction and environment

Weights and source checkout are outside Git at
`C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1`. The source was
checked out at the pinned commit above; the release asset SHA and byte count are
recorded here and in `R1_CANDIDATE_INVENTORY.json`. Runtime used the existing
isolated audit dependency directory, Python 3.11, PyTorch/torchaudio
`2.6.0+cu124`, and the local RTX 3080 Laptop GPU. No global package was
installed, no data benchmark was loaded, and no cloud compute was used.

```powershell
$env:PYTHONPATH = 'C:\CodexArtifacts\pretrained-reuse-r1\python-deps;C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\source'
@'
import torch
from models.dymn.model import get_model
from models.preprocess import AugmentMelSTFT

model = get_model(num_classes=527, width_mult=0.4)
state = torch.load(r'C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\dymn04_as.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
preprocess = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, freqm=0, timem=0).eval()
wave = torch.sin(2 * torch.pi * 440 * torch.arange(64000) / 32000)[None, :] * 0.1
with torch.inference_mode():
    logits, pooled = model(preprocess(wave).unsqueeze(0))
print(logits.shape, pooled.shape, torch.isfinite(logits).all().item())
'@ | py -3.11 -
```

## Sources

- [EfficientAT pinned source README](https://github.com/fschmid56/EfficientAT/tree/a425fdce92572e602a1d5634799bd9f1f2efa806)
- [EfficientAT release asset and byte size](https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1)
- [EfficientAT checkpoint commercial permission, owner response](https://github.com/fschmid56/EfficientAT/issues/44)
- [ViT-Lens upstream release statement](https://github.com/TencentARC/ViT-Lens)
- [Meta PE Core upstream model/checkpoint docs](https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/apps/pe/README.md)
- [VATT official checkpoint list](https://github.com/google-research/google-research/blob/5b381db4139c3cbf25cc9a698c5fe9754215fae2/vatt/README.md)
- [MJEPA official repo](https://github.com/facebookresearch/MJEPA/tree/82f3472219f2f58672999ed654314b34ad590c07), [release list](https://github.com/facebookresearch/MJEPA/releases)
