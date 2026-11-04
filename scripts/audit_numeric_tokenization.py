"""Read-only contextual numeral tokenization audit for fixed training corpora."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def numeric_spans(text):
    return [m.span(1) for m in re.finditer(r'"(?:cost|risk|time)"\s*:\s*(\d+)', text)]


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reports = {}
    expected = {
        "conditional-curriculum-01": (
            "4d6110f4a401cfd6e50b9d0720a3a5fcc3aea146118ff7d1b0a05351764afbbe"
        ),
        "numeric-range-curriculum-01": (
            "95f4edc33711a6ac334fff77cb08300a45bbe2bc8acefad1e8591b069c076101"
        ),
    }
    for name, digest in expected.items():
        path = args.root / name / "train.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("corpus hash mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        states = {}
        for row in rows:
            if row["group"] in states and states[row["group"]]["state"] != row["state"]:
                raise ValueError("inconsistent underlying state")
            states[row["group"]] = row
        counts, by_scale = Counter(), {}
        numbers = set()
        unknown = 0
        for row in states.values():
            text = "State: " + row["state"]
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            spans = numeric_spans(text)
            if len(spans) != 12:
                raise ValueError("unexpected numeric field count")
            scale = row.get("numeric_transform", {}).get("scale", 1)
            counter = by_scale.setdefault(scale, Counter())
            for start, end in spans:
                ids = [
                    i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start
                ]
                if not ids:
                    raise ValueError("unrepresented numeric value")
                counts[len(ids)] += 1
                counter[len(ids)] += 1
                numbers.add(int(text[start:end]))
                unknown += sum(encoded["input_ids"][i] == tokenizer.unk_token_id for i in ids)
        reports[name] = {
            "source_sha256": digest,
            "states": len(states),
            "occurrences": sum(counts.values()),
            "unique_numeric_values": len(numbers),
            "contextual_token_counts": dict(counts),
            "by_scale": {str(s): dict(c) for s, c in by_scale.items()},
            "multi_token_fraction": sum(v for k, v in counts.items() if k > 1)
            / sum(counts.values()),
            "unknown_token_occurrences": unknown,
        }
    result = {
        "reports": reports,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.model.iterdir()
            if p.is_file() and ("token" in p.name or "vocab" in p.name or p.name == "config.json")
        },
        "limitation": "Magnitude and vocabulary change together; not causal proof.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
