"""Complete immutable spatial corpus authority and memory-bounded input routing."""

import json
from pathlib import Path

import numpy as np

from scripts.extract_spatial_context_corpus import sha
from tiny_omni_decision.spatial_context_features import spatial_context_positions

SPATIAL_RESULT_PIN = "a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da"


def load_spatial_bank(result_path, plans, expected_sha256):
    result_path = Path(result_path)
    if sha(result_path) != expected_sha256:
        raise ValueError("spatial result hash changed")
    result = json.loads(result_path.read_text())
    if (
        result["status"] != "PASS_COMPLETE_SPATIAL_CONTEXT_CORPUS_NOT_QUALITY"
        or result["smoke"]
        or result["state_before"] != result["state_after"]
        or not result["first_repeat_bitwise"]
        or result["training"]
        or result["annotations_loaded"]
        or result["sealed_audit_opened"]
        or result["parameters"] != 31072327
    ):
        raise ValueError("complete frozen corpus mandatory")
    config_path = result_path.parent / "config.json"
    if sha(config_path) != result["config_sha256"]:
        raise ValueError("spatial effective config changed")
    config = json.loads(config_path.read_text())
    if (
        config["plans"] != plans
        or config["smoke"]
        or (config["frames"], config["grid_size"]) != (32, 16)
    ):
        raise ValueError("spatial plans/geometry changed")
    hashes = {str(result_path): expected_sha256, str(config_path): result["config_sha256"]}
    expected = {p["video_id"]: p for p in plans}
    if len(expected) != len(plans):
        raise ValueError("duplicate expected identity")
    bank, rgb = {}, {"train": set(), "valid": set()}
    for row in result["features"]:
        vid = row["video_id"]
        plan = expected.get(vid)
        if (
            vid in bank
            or plan is None
            or row["split"] != plan["split"]
            or row["media_sha256"] != plan["media_sha256"]
            or row["original_frames"] != plan["frames"]
            or row["frame_indices"] != plan["frame_indices"]
            or not row["rng_unchanged"]
        ):
            raise ValueError("spatial identity/media/time binding changed")
        if set(row["outputs"]) != {"appearance", "positions", "frames"}:
            raise ValueError("spatial outputs incomplete")
        expected_arrays = {
            "appearance": ((32, 256, 384), "float32"),
            "positions": ((32, 256, 3), "float32"),
            "frames": ((32,), "int64"),
        }
        for name, (shape, dtype) in expected_arrays.items():
            info = row["outputs"][name]
            path = Path(info["path"])
            if sha(path) != info["sha256"] or path.stat().st_size != info["file_bytes"]:
                raise ValueError("spatial output file changed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                array.shape != shape
                or str(array.dtype) != dtype
                or info["shape"] != list(shape)
                or info["dtype"] != dtype
                or not np.isfinite(array).all()
            ):
                raise ValueError("invalid spatial array")
            if name == "frames" and not np.array_equal(array, plan["frame_indices"]):
                raise ValueError("spatial frame routing changed")
            if name == "positions" and not np.array_equal(
                array,
                spatial_context_positions(plan["frame_indices"], source_frames=plan["frames"]),
            ):
                raise ValueError("spatial positions changed")
            hashes[str(path)] = info["sha256"]
        bank[vid] = row
        rgb[row["split"]].add(row["selected_rgb_sha256"])
    if set(bank) != set(expected) or rgb["train"] & rgb["valid"]:
        raise ValueError("incomplete spatial corpus or cross-split RGB overlap")
    return bank, hashes


def context_tokens(row, arm, control="normal"):
    if arm not in {"initial_broadcast", "temporal_appearance"}:
        raise ValueError("unknown appearance arm")
    if control not in {"normal", "zero", "mismatch", "initial", "reverse"}:
        raise ValueError("unknown context control")
    appearance = np.load(row["outputs"]["appearance"]["path"], mmap_mode="r", allow_pickle=False)
    positions = np.load(row["outputs"]["positions"]["path"], mmap_mode="r", allow_pickle=False)
    if arm == "initial_broadcast" or control == "initial":
        appearance = np.broadcast_to(appearance[:1], appearance.shape)
    elif control == "reverse":
        appearance = appearance[::-1]
    combined = np.concatenate((appearance, positions), axis=-1).reshape(8192, 387)
    if control == "zero":
        combined.fill(0)
    return combined
