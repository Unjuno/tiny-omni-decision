"""Prospectively fixed long curve; no outcome-dependent stop or budget extension."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

MAX_STEPS = 32768
EVALUATION_STEPS = (0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768)


def learning_rate(step):
    if type(step) is not int or not 1 <= step <= MAX_STEPS:
        raise ValueError("fixed32768-step schedule required")
    if step <= 62:
        return 0.001 * step / 62
    return 0.0005 * (1 + math.cos(math.pi * (step - 62) / (MAX_STEPS - 62)))


def extend_plan(pool, batches, max_steps=MAX_STEPS):
    if (
        not pool
        or not batches
        or type(max_steps) is not int
        or max_steps < len(batches)
        or max_steps % len(batches)
        or any(type(i) is not int or i < 0 for i in pool)
        or any(len(b) != 8 for b in batches)
        or any(type(o) is not int for b in batches for o in b)
        or sorted(o for b in batches for o in b) != list(range(len(pool)))
    ):
        raise ValueError("complete eight-microbatch occurrence plan and whole cycles required")
    cycles = max_steps // len(batches)
    order = [[pool[o] for o in b] for _ in range(cycles) for b in batches]
    occurrences = [[o + cycle * len(pool) for o in b] for cycle in range(cycles) for b in batches]
    payload = dict(order=order, occurrences=occurrences)
    return dict(
        **payload,
        cycles=cycles,
        unique_row_indices=sorted(set(pool)),
        order_sha256=hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def fixed_parent_probe_indices(long_plan, parent_pool):
    """Return the shared original-pool TRAIN probe for repeat and fresh plans."""
    unique = long_plan.get("unique_row_indices") if isinstance(long_plan, dict) else None
    if (
        not isinstance(unique, list)
        or not unique
        or any(type(index) is not int or index < 0 for index in unique)
        or unique != sorted(set(unique))
        or not parent_pool
        or any(type(index) is not int or index < 0 for index in parent_pool)
    ):
        raise ValueError("complete unique long-plan rows and valid parent pool required")
    parent = set(parent_pool)
    if not parent.issubset(unique):
        raise ValueError("long plan must retain every original-pool question")
    probe = [index for index in unique if index in parent]
    if len(probe) != len(parent):
        raise ValueError("shared parent probe must contain each original row once")
    return probe


def verify_pins(pins, source_commits=(), repo_root=None):
    """Verify files directly, falling back to an exact historical Git blob."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    verified = {}
    for name, expected in pins.items():
        path = Path(name)
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual == expected:
                verified[name] = {"method": "file_sha256", "sha256": actual}
                continue
        normalized = name.replace("/", "\\")
        folded = normalized.casefold()
        marker = "\\pretrained-reuse-omni\\"
        if marker not in folded:
            raise ValueError(f"missing or changed pinned input: {name}")
        relative_path = normalized[folded.index(marker) + len(marker) :].replace("\\", "/")
        for commit in source_commits:
            try:
                content = subprocess.check_output(
                    ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                continue
            actual = hashlib.sha256(content).hexdigest()
            if actual == expected:
                verified[name] = {
                    "method": "pinned_git_blob_sha256",
                    "commit": commit,
                    "path": relative_path,
                    "sha256": actual,
                }
                break
        else:
            raise ValueError(f"pinned source differs from receipt and Git commits: {name}")
    return verified
