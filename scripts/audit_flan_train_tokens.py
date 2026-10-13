"""Train-only tokenizer and oracle audit; no model scoring or data mutation."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

source = Path("C:/CodexArtifacts/pretrained-reuse-r2/context-robustness-corpus-01/train.jsonl")
assert (
    hashlib.sha256(source.read_bytes()).hexdigest()
    == "a0116e4e2c5024fe39e113461c3b286da3d76b1f8b11940dc195fd5594c2982b"
)
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
tokenizer = AutoTokenizer.from_pretrained(
    "C:/CodexArtifacts/pretrained-reuse-r1/flan-t5-small-runtime-01", local_files_only=True
)
texts = []
correct = 0
for r in rows:
    state = json.loads(r["state"])
    rule = r["oracle"]
    op = min if rule["direction"] == "minimum" else max
    assert rule["direction"] in ("minimum", "maximum")
    expected = op(state, key=lambda x: x[rule["field"]])["id"]
    assert expected == r["target"] and expected in r["options"]
    correct += 1
    texts.append(
        "State: "
        + r["state"]
        + "\nCondition: "
        + r["question"]
        + "\n"
        + "\n".join(f"{chr(65 + i)}: {o}" for i, o in enumerate(r["options"]))
        + "\nAnswer with the option letter."
    )
encoded = tokenizer(texts, truncation=False)["input_ids"]
without = tokenizer([t.replace("{", "").replace("}", "") for t in texts], truncation=False)[
    "input_ids"
]
result = dict(
    train_records=len(rows),
    oracle_verified=correct,
    unknown_histogram=dict(Counter(x.count(tokenizer.unk_token_id) for x in encoded)),
    unknown_without_braces_histogram=dict(
        Counter(x.count(tokenizer.unk_token_id) for x in without)
    ),
    max_tokens=max(map(len, encoded)),
    truncated_records=sum(len(x) > 512 for x in encoded),
    distinct_prompt_count=len(set(texts)),
    distinct_token_sequences=len({tuple(x) for x in encoded}),
    labels={c: tokenizer.encode(c, add_special_tokens=False) for c in "ABCD"},
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    inference=False,
    modification=False,
)
out = Path("C:/CodexArtifacts/pretrained-reuse-r2/flan-tokenizer-train-audit-01")
out.mkdir(exist_ok=False)
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
Path("docs/pretrained_reuse/R2_FLAN_TOKENIZER_AUDIT_20261013.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
