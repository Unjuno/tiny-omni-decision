"""Actual fine-grid spatial inputs; coarse features are never upsampled."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tiny_omni_decision.spatial_context_features import pooled_spatial_context


def fine_context(low, high):
    if (
        not isinstance(low, np.ndarray)
        or not isinstance(high, np.ndarray)
        or low.ndim != 4
        or high.ndim != 4
        or low.shape[1:] != (32, 32, 256)
        or high.shape[1:] != (64, 64, 128)
        or low.dtype != np.float32
        or high.dtype != np.float32
    ):
        raise ValueError("native FP32 low32/high64 feature grids required")
    return pooled_spatial_context(low, high, grid_size=32)


def fine_array_bytes(clips):
    if type(clips) is not int or clips < 1:
        raise ValueError("positive integer clip count required")
    return clips * (32 * 1024 * (384 + 3) * 4 + 32 * 8 + 3 * 128)


def require_disk(free_bytes, clips):
    if free_bytes < fine_array_bytes(clips) + 8 * 2**30 + 64 * 2**20:
        raise OSError("fine corpus would breach protected8GiB reserve")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_pinned_file(path, expected):
    if file_sha(path) != expected:
        raise ValueError("historically pinned source file changed")


def require_reference(directory, contract_sha):
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(json.dumps(config["contract"], sort_keys=True).encode()).hexdigest()
    if (
        report["status"] != "PASS_ALL_TRAIN_FINE_SPATIAL_CORPUS_NOT_QUALITY"
        or file_sha(directory / "config.json") != report["config_sha256"]
        or actual != contract_sha
        or config["contract_sha256"] != contract_sha
        or report["contract_sha256"] != contract_sha
        or config["smoke"] is not False
        or len(report["features"]) != 99
        or len({r["video_id"] for r in report["features"]}) != 99
        or config["selected_train_ids"] != [r["video_id"] for r in report["features"]]
        or config["selected_train_ids"] != [p["video_id"] for p in config["contract"]["plans"]]
    ):
        raise ValueError("complete same-actual-contract main reference required")
    return report


def require_smoke(directory, contract_sha):
    report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    actual_contract = hashlib.sha256(
        json.dumps(config["contract"], sort_keys=True).encode()
    ).hexdigest()
    if (
        report["status"] != "PASS_TRAIN_FINE_SPATIAL_SMOKE_NOT_QUALITY"
        or report["contract_sha256"] != contract_sha
        or file_sha(directory / "config.json") != report["config_sha256"]
        or report["parameters"] != 31072327
        or not report["first_repeat_bitwise"]
        or report["state_before"] != report["state_after"]
        or report["state_before"] != config["contract"]["state_sha256"]
        or actual_contract != contract_sha
        or config["contract_sha256"] != contract_sha
        or config["smoke"] is not True
        or config["selected_train_ids"] != ["video_10103"]
        or len(report["features"]) != 1
        or report["features"][0]["video_id"] != "video_10103"
        or not report["features"][0]["coarse_reference_bitwise"]
    ):
        raise ValueError("complete same-contract train-only smoke required")
    outputs = report["features"][0]["outputs"]
    if set(outputs) != {"appearance", "positions", "frames"}:
        raise ValueError("complete saved smoke array inventory required")
    for info in outputs.values():
        if file_sha(info["path"]) != info["sha256"]:
            raise ValueError("protected smoke array changed")
    return report
