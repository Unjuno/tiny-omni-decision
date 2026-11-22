from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.uci_har_validation_analysis import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze saved UCI HAR validation predictions without loading a model."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_run(args.run_dir, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": result["comparison"]["count"],
                "validation_subjects": result["validation_subjects"],
                "static_accuracy": result["comparison"]["static"]["accuracy"],
                "lstm_accuracy": result["comparison"]["lstm"]["accuracy"],
                "official_test_opened": result["official_test_opened"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
