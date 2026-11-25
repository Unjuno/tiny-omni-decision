"""Create the metadata-only RoboPro prompt-group development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.robopro_split import build_from_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    args = parser.parse_args()

    manifest = build_from_paths(
        args.info,
        args.tasks,
        args.episodes,
        seed=args.seed,
        development_fraction=args.development_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_episodes": manifest["train"]["episode_count"],
                "development_episodes": manifest["development"]["episode_count"],
                "train_prompt_groups": manifest["train"]["prompt_group_count"],
                "development_prompt_groups": manifest["development"]["prompt_group_count"],
                "source_files": manifest["source_files"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
