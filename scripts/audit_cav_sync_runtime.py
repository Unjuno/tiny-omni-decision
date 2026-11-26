"""CPU pretrained tensor-load and deterministic forward audit; no quality claim."""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

root = Path("C:/CodexArtifacts/pretrained-reuse-r1")
source = root / "cav-mae-sync-source-audit-01"
out = root / "cav-mae-sync-runtime-01"
out.mkdir(exist_ok=False)
sys.path.insert(0, str(root / "cav-mae-sync-deps-01"))
package = types.ModuleType("cav_structure")
package.__path__ = [str(source)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    "cav_structure.model", source / "src__models__cav_mae_sync.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
torch.manual_seed(17)
model = (
    module.CAVMAE(
        audio_length=416,
        modality_specific_depth=11,
        num_register_tokens=8,
        cls_token=True,
        total_frame=4,
        contrastive_heads=False,
    )
    .eval()
    .requires_grad_(False)
)
checkpoint = root / "cav-mae-sync-weight-audit-01/checkpoint.pth"
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
    "a035b04cf615fe4c85d6aa1e74b26d8223a85945957664e66abea16e183629bd"
)
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
assert all(k.startswith("module.") for k in state)
model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
del state
used = set()
handles = []
for name, child in model.named_modules():
    if list(child.parameters(recurse=False)):
        handles.append(child.register_forward_hook(lambda _m, _i, _o, name=name: used.add(name)))
with torch.inference_mode():
    result = model.forward_feat(torch.zeros(2, 416, 128), torch.zeros(2, 3, 224, 224))
with torch.inference_mode():
    repeated = model.forward_feat(torch.zeros(2, 416, 128), torch.zeros(2, 3, 224, 224))
assert all(torch.equal(a, b) for a, b in zip(result, repeated, strict=True))
for handle in handles:
    handle.remove()
parameters = {name: p.numel() for name, p in model.named_parameters()}
# Direct root parameters are used without child forward calls.
roots = {
    "modality_a",
    "modality_v",
    "pos_embed_a",
    "pos_embed_v",
    "register_tokens",
    "cls_token_a",
    "cls_token_v",
}
active = {
    name: n for name, n in parameters.items() if name in roots or name.rsplit(".", 1)[0] in used
}
report = {
    "status": "PRETRAINED_CPU_ZERO_INPUT_SMOKE_ONLY",
    "checkpoint_loaded": True,
    "strict_load": True,
    "exact_repeat": True,
    "audio_length": 416,
    "torch": torch.__version__,
    "timm": sys.modules["timm"].__version__,
    "dependency_note": "Isolated upstream-requested timm 0.4.5; no environment upgrade.",
    "full_instantiated_parameters": sum(parameters.values()),
    "forward_feat_used_parameters": sum(active.values()),
    "used_plus_existing_text_parameters": sum(active.values()) + 22713601,
    "output_shapes": [list(x.shape) for x in result],
    "outputs_finite": all(torch.isfinite(x).all().item() for x in result),
    "parameters": parameters,
    "used_parameters": active,
    "source_hashes": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source / "src__models__cav_mae_sync.py", source / "pos_embed.py")
    },
}
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ("parameters", "used_parameters")}, indent=2
    )
)
