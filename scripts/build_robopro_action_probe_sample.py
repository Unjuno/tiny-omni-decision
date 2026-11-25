"""Freeze a seed-17 metadata-only RoboPro action-probe sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.robopro_probe_sample import build_probe_sample


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-episode-ids", type=int, nargs="*", default=[])
    parser.add_argument("--prompts-per-split", type=int, default=16)
    parser.add_argument("--episodes-per-prompt", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen sample: {args.output}")
    episodes = _read_jsonl(args.episodes.resolve(strict=True))
    split_manifest_path = args.split_manifest.resolve(strict=True)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    sample = build_probe_sample(
        episodes,
        split_manifest,
        prompts_per_split=args.prompts_per_split,
        episodes_per_prompt=args.episodes_per_prompt,
        excluded_episode_ids=set(args.exclude_episode_ids),
    )
    sample["source_metadata_sha256"] = hashlib.sha256(args.episodes.read_bytes()).hexdigest()
    sample["split_manifest_sha256"] = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    payload = (json.dumps(sample, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(
        json.dumps(
            {
                "status": "sample_frozen",
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "train_episodes": sample["train_episode_count"],
                "development_episodes": sample["development_episode_count"],
                "train_prompt_groups": sample["train_prompt_groups"],
                "development_prompt_groups": sample["development_prompt_groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
