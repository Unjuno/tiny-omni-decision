"""Compute paired, document-family bootstrap diagnostics from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.openjev_amount_probe import paired_family_bootstrap


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = paired_family_bootstrap(
        _read_jsonl(args.reference),
        _read_jsonl(args.candidate),
        seed=args.seed,
        iterations=args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
