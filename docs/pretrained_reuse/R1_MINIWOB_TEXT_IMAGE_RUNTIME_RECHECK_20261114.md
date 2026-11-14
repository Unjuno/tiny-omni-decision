# R1 MiniWoB text/image encoder runtime recheck — 2026-11-14

## Scope

Read-only, offline CPU feasibility recheck for a prospective MiniWoB
criterion-conditioned Path A. This confirms local checkpoint identity, loading,
finite feature outputs, and frozen parameters. It is not a task evaluation,
training run, or claim of general text/image capability. No benchmark or sealed
audit data was opened.

Repository source: `codex/r2-miniwob-criterion-conditioned-r0refresh-20261114`
at `0fc3f53a` (runtime implementation cherry-picked onto latest R0/R2 progress
commit `932e962c`).

## Pinned local checkpoints

| Encoder | Revision | Local safetensors SHA-256 | File bytes | Parameters | Rights evidence |
|---|---|---|---:|---:|---|
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | `4ca70771034acceecb2e72475f72050fcdde4ddc` | `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` | 69,569,488 | 17,389,824 | Pinned Hugging Face model card declares Apache-2.0. |
| `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` | 88,249,960 | 22,056,576 | Pinned model card declares Apache-2.0; upstream DINOv2 README says code and model weights use Apache-2.0. |

The two encoders total **39,446,400 resident parameters** and **157,819,448
weight-file bytes**. This is only a text/image sub-stack; it excludes readout,
audio, video, runtime dependencies, and package overhead. It does not establish
that a complete four-modality model fits the product budget.

## Runtime procedure and result

Using Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, and safetensors
0.8.0, a separate process loaded both directories with `local_files_only=True`
and `trust_remote_code=False` on CPU. No packages, checkpoints, or source files
were changed. The RTX 3080 was not used.

- MiniLM encoded two fixed prompts with local tokenization and attention-mask
  mean pooling; output shape `[2, 384]`, all finite.
- DINOv2 encoded a locally generated synthetic 160x210 RGB screen after the
  Path A 224x224 bicubic resize and RGB normalization; output shape
  `[1, 257, 384]`, all finite.
- Every parameter in both models had `requires_grad=False`.
- Combined load and forward wall time was 2.354 seconds in this CPU process; it
  is a smoke measurement, not a stable latency benchmark.

The computed safetensors hashes matched the repository's previously pinned
hashes. No decision head, calibrated probabilities, quality metrics, or package
size were measured here. The text pooling implementation must be frozen before
any future comparison; this smoke does not select it for product use.

## Reproduction

Run from PowerShell with the local model directories above and the already
installed Python environment:

```powershell
@'
import torch
from transformers import AutoModel, AutoTokenizer

text_dir = r'C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en'
vision_dir = r'C:\CodexArtifacts\pretrained-reuse-r1\dinov2-small-audit-01\model'
tokenizer = AutoTokenizer.from_pretrained(text_dir, local_files_only=True, trust_remote_code=False)
text_model = AutoModel.from_pretrained(text_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
vision_model = AutoModel.from_pretrained(vision_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
tokens = tokenizer(['Select the red button.', 'Select the blue button.'], padding=True, return_tensors='pt')
with torch.inference_mode():
    hidden = text_model(**tokens).last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
    text_features = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    # The recorded image smoke used the DINO 224x224 resize/normalization path.
    from PIL import Image
    import numpy as np
    image = Image.new('RGB', (160, 210), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for x, color in zip((15, 50, 85, 120), ('red', 'blue', 'green', 'yellow')):
        draw.rectangle((x, 80, x + 22, 110), fill=color)
    pixels = np.asarray(image.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    pixel_values = torch.from_numpy(((pixels - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0)
    image_features = vision_model(pixel_values=pixel_values).last_hidden_state
print(text_features.shape, image_features.shape)
print(sum(p.numel() for p in text_model.parameters()), sum(p.numel() for p in vision_model.parameters()))
'@ | py -3.11 -
```

The full audited process also verified hashes and frozen status; those values
are recorded above and in the command-session output. A dedicated audit CLI has
not yet been added.

## Rights references

- [Pinned MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc)
- [Pinned DINOv2-Small model card](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
- [DINOv2 upstream README and license statement](https://github.com/facebookresearch/dinov2/blob/main/README.md#license)

## Decision

**Technical runtime smoke: PASS.** These two pinned encoders are locally
loadable together with finite features and are well below 200M parameters as a
partial sub-stack. **Path A quality, four-modality budget, end-to-end package,
and deployment gates remain OPEN.** Continue only with a fresh, grouped,
leakage-checked criterion-conditioned corpus and a frozen evaluation protocol.
