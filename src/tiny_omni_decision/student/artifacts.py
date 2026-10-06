"""Checksummed packed exports; reference reload deliberately dequantizes in PyTorch.

These artifacts are not optimizer/resume checkpoints and are not GGUF files.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .data import file_hash, write_json
from .packing import pack_trits, unpack_trits
from .ternary import TernaryController, dequantize, owner_attribute

FORMAT = "tiny-omni-packed-ternary-v1"


def bundle_info(path: Path) -> dict[str, Any]:
    info = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if info.get("format") != FORMAT:
        raise ValueError("unsupported packed artifact format")
    if file_hash(path / "model.safetensors") != info["weights_sha256"]:
        raise ValueError("packed weight checksum mismatch")
    return info


def save_bundle(c: TernaryController, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".student-export-", dir=path.parent))
    tensors = {}
    entries = {}
    byte_counts = dict(code_bytes=0, scale_bytes=0, exception_bytes=0, adapter_bytes=0)
    try:
        for index, name in enumerate(c.state_keys):
            owner, attribute = owner_attribute(c.model, name)
            value = getattr(owner, attribute)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"unsupported non-tensor state: {name}")
            prefix = f"t{index}"
            row = {"shape": list(value.shape), "dtype": str(value.dtype), "prefix": prefix}
            if name in c.targets:
                codes, scales = c.components(name)
                packed = pack_trits(codes).cpu().contiguous()
                scales = scales.float().cpu().contiguous()
                tensors[prefix + ".codes"] = packed
                tensors[prefix + ".scales"] = scales
                row["kind"] = "ternary"
                byte_counts["code_bytes"] += packed.numel()
                byte_counts["scale_bytes"] += scales.numel() * scales.element_size()
                if name in c.loras:
                    adapter = c.loras[name]
                    row["lora"] = {"rank": adapter.rank, "alpha": adapter.alpha}
                    for suffix in ("lora_A", "lora_B"):
                        factor = getattr(adapter, suffix).detach().cpu().contiguous().clone()
                        if not torch.isfinite(factor).all():
                            raise ValueError("non-finite LoRA")
                        tensors[prefix + "." + suffix] = factor
                        byte_counts["adapter_bytes"] += factor.numel() * factor.element_size()
            else:
                value = value.detach().cpu().contiguous().clone()
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"non-finite exception tensor: {name}")
                row["kind"] = "dense"
                tensors[prefix + ".dense"] = value
                byte_counts["exception_bytes"] += value.numel() * value.element_size()
            entries[name] = row
        save_file(tensors, temp / "model.safetensors")
        info = {
            "format": FORMAT, "packing": "five_trits_per_byte", "code_bits_per_weight": 1.6,
            "group_size": c.group_size, "parameter_count": c.parameter_count,
            "entries": entries, "metadata": metadata, "byte_counts": byte_counts,
            "weights_sha256": file_hash(temp / "model.safetensors"),
            "execution": "reference_dequantized_float_no_packed_kernel",
        }
        write_json(temp / "manifest.json", info)
        # Destination is never replaced. Renaming a completed directory is atomic on the same FS.
        if path.exists():
            raise FileExistsError(path)
        temp.rename(path)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    total = sum(p.stat().st_size for p in path.iterdir())
    return byte_counts | {"total_bytes": total,
                         "whole_artifact_bpw": total * 8 / c.parameter_count}


def load_bundle(c: TernaryController, path: Path) -> dict[str, Any]:
    info = bundle_info(path)
    rows = info["entries"]
    if info["group_size"] != c.group_size or set(rows) != set(c.state_keys):
        raise ValueError("artifact/model inventory mismatch")
    if {n for n, r in rows.items() if r["kind"] == "ternary"} != set(c.targets):
        raise ValueError("artifact quantization target mismatch")
    tensors = load_file(path / "model.safetensors", device="cpu")
    expected_keys = set()
    decoded = {}
    lora_rows = {name: row["lora"] for name, row in rows.items() if "lora" in row}
    if c.loras and set(c.loras) != set(lora_rows):
        raise ValueError("existing adapter inventory differs from artifact")
    # Validate the complete payload before mutating the model.
    for name, row in rows.items():
        owner, attribute = owner_attribute(c.model, name)
        reference = getattr(owner, attribute)
        if list(reference.shape) != row["shape"] or str(reference.dtype) != row["dtype"]:
            raise ValueError(f"shape/dtype mismatch: {name}")
        prefix = row["prefix"]
        if row["kind"] == "ternary":
            keys = {prefix + ".codes", prefix + ".scales"}
            scales = tensors[prefix + ".scales"]
            if scales.dtype != torch.float32:
                raise ValueError("scales must be FP32")
            codes = unpack_trits(tensors[prefix + ".codes"], reference.numel()).reshape(
                reference.shape
            )
            dequantize(codes, scales, c.group_size)
            decoded[name] = (codes, scales)
            if "lora" in row:
                spec = row["lora"]
                if type(spec["rank"]) is not int or not 0 < spec["rank"] <= min(reference.shape):
                    raise ValueError("invalid adapter rank")
                for suffix, shape in (
                    ("lora_A", (spec["rank"], reference.shape[1])),
                    ("lora_B", (reference.shape[0], spec["rank"])),
                ):
                    value = tensors[prefix + "." + suffix]
                    if tuple(value.shape) != shape or value.dtype != reference.dtype:
                        raise ValueError("invalid adapter shape/dtype")
                    if not torch.isfinite(value).all():
                        raise ValueError("non-finite adapter")
                    keys.add(prefix + "." + suffix)
        elif row["kind"] == "dense":
            keys = {prefix + ".dense"}
            value = tensors[prefix + ".dense"]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError("invalid dense tensor shape/dtype")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("non-finite dense tensor")
        else:
            raise ValueError("invalid artifact entry")
        expected_keys |= keys
    if expected_keys != set(tensors):
        raise ValueError("unexpected or missing artifact tensors")
    with torch.no_grad():
        for name, row in rows.items():
            owner, attribute = owner_attribute(c.model, name)
            reference = getattr(owner, attribute)
            if name in decoded:
                codes, scales = decoded[name]
                c.targets[name].set_frozen(codes.to(reference.device), scales.to(reference.device))
            else:
                reference.copy_(tensors[row["prefix"] + ".dense"].to(reference.device))
    c.freeze()
    if lora_rows and not c.loras:
        alphas = {spec["alpha"] for spec in lora_rows.values()}
        if len(alphas) != 1:
            raise ValueError("mixed LoRA alpha is unsupported")
        c.add_lora(max(spec["rank"] for spec in lora_rows.values()),
                   next(iter(alphas)), targets=list(lora_rows))
    with torch.no_grad():
        for name, adapter in c.loras.items():
            row = rows[name]
            if adapter.rank != row["lora"]["rank"] or adapter.alpha != row["lora"]["alpha"]:
                raise ValueError("LoRA configuration mismatch")
            for suffix in ("lora_A", "lora_B"):
                factor = getattr(adapter, suffix)
                factor.copy_(tensors[row["prefix"] + "." + suffix].to(factor.device))
    c.model.requires_grad_(False)
    c.model.eval()
    return info
