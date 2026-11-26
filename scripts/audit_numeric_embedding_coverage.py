"""Training-only numeric-token coverage before changing embedding trainability."""

import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    path = root / "comparison-vocabulary-curriculum-01/train.jsonl"
    assert sha(path) == "41b35f3c2b1161d21ebf123d8ba39b29aa17dc9dc82c5f03ff7f291b9d368c76"
    out = root / "numeric-embedding-coverage-01"
    if out.exists():
        raise FileExistsError(out)
    model = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    assert tokenizer.is_fast
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    number_counts = Counter()
    token_counts = Counter()
    lengths = Counter()
    used_states = set()
    for row in rows:
        if row["state"] in used_states:
            continue
        used_states.add(row["state"])
        text = row["state"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for match in re.finditer(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])", text):
            ids = [
                tid
                for tid, (a, b) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
                if a < match.end() and b > match.start()
            ]
            assert ids
            number_counts[match.group()] += 1
            token_counts.update(ids)
            lengths[len(ids)] += 1
    result = dict(
        scope="all unique training states only; not actual scheduled/selected exposure",
        corpus_rows=len(rows),
        unique_states=len(used_states),
        numeric_occurrences=sum(lengths.values()),
        numeric_values=len(number_counts),
        token_lengths=dict(lengths),
        numeric_subword_ids=len(token_counts),
        potential_delta_parameters=len(token_counts) * 384,
        full_word_embedding_parameters=30522 * 384,
        numeric_tokens=[
            dict(id=k, token=tokenizer.convert_ids_to_tokens(k), count=v)
            for k, v in sorted(token_counts.items())
        ],
        input_sha256=sha(path),
        tokenizer_sha256=sha(model / "tokenizer.json"),
        existing_policy="all6 encoder blocks + classifier; word embeddings frozen",
        causal_claim=False,
    )
    out.mkdir()
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R2_NUMERIC_EMBEDDING_COVERAGE_20261012.json"), result)
    print({k: v for k, v in result.items() if k != "numeric_tokens"})


if __name__ == "__main__":
    main()
