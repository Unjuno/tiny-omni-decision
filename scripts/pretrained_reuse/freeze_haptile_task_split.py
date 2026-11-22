"""Freeze a task-held-out split from the pinned HapTile conversion card only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.haptile_split import (  # noqa: E402
    CONVERSION_CARD_SHA256,
    build_haptile_task_split,
)

TASK_ROW = re.compile(r"^\| `(?P<task>[a-z0-9_]+)` \| (?P<count>\d+) \|")


def parse_task_counts(card_path: Path) -> tuple[dict[str, int], str]:
    card_bytes = card_path.read_bytes()
    card_sha256 = hashlib.sha256(card_bytes).hexdigest()
    if card_sha256 != CONVERSION_CARD_SHA256:
        raise ValueError("conversion README does not match the pinned card hash")
    task_counts: dict[str, int] = {}
    for line in card_bytes.decode("utf-8").splitlines():
        match = TASK_ROW.match(line)
        if match:
            task = match.group("task")
            if task in task_counts:
                raise ValueError(f"duplicate task row in pinned card: {task}")
            task_counts[task] = int(match.group("count"))
    return task_counts, card_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    if args.public_output.exists() or args.private_output.exists():
        raise SystemExit("refusing to overwrite an existing split manifest")

    counts, card_sha256 = parse_task_counts(args.card)
    public_payload, private_payload = build_haptile_task_split(
        counts, conversion_card_sha256=card_sha256
    )
    for path, payload in (
        (args.public_output, public_payload),
        (args.private_output, private_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "public_manifest": str(args.public_output),
        "private_manifest": str(args.private_output),
        "public_manifest_sha256": public_payload["manifest_sha256"],
        "private_manifest_sha256": private_payload["manifest_sha256"],
        "development_episode_counts": public_payload["development_episode_counts"],
        "sealed_audit_group_count": public_payload["sealed_audit"]["task_group_count"],
        "sealed_audit_episode_count": public_payload["sealed_audit"]["episode_count"],
        "sealed_audit_group_commitment_sha256": (
            public_payload["sealed_audit"]["group_and_count_commitment_sha256"]
        ),
        "sealed_audit_task_ids_emitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
