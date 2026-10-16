"""Independent data/transport diagnostics; never rewrite a corpus or select a model."""

import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np


def pair_digest(question, option):
    value = json.dumps([question, option], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def validate_associations(rows, observations, pairs):
    """Check raw option strings against actual query keys, including reordered labels."""
    seen, inputs, conflicts = set(), {}, []
    for row in rows:
        options = row["options"]
        if (
            row["id"] in seen or row["split"] not in {"train", "validation"}
            or row["split"] != observations[row["observation"]]["split"]
            or not 2 <= len(options) <= 40 or len(set(options)) != len(options)
            or len(row["option_ids"]) != len(options)
            or len(set(row["option_ids"])) != len(options)
            or type(row["target_index"]) is not int
            or not 0 <= row["target_index"] < len(options)
            or row["queries"] != [pair_digest(row["question"], opt) for opt in options]
        ):
            raise ValueError("invalid row/query/target/split association")
        seen.add(row["id"])
        for key, opt in zip(row["queries"], options, strict=True):
            if pairs[key] != [row["question"], opt]:
                raise ValueError("query pair lookup disagrees with raw input")
        key = (row["observation"], row["question"], tuple(sorted(options)))
        target = options[row["target_index"]]
        old = inputs.setdefault(key, (row["id"], target))
        if old[1] != target:
            conflicts.append([old[0], row["id"]])
    return dict(rows=len(seen), conflicting_identical_inputs=conflicts)


def semantic_text_target(row):
    """Recompute supported labels without importing any generating/oracle function.

    Returns a description in the supplied candidate vocabulary. Open-ended legacy
    annotations are explicitly unverified, not accepted by an imitation oracle.
    """
    source = row["source"]
    if source == "original":
        return None
    state = json.loads(row["state"])
    if source in {"condition_relevance_v1", "binding"} or (
        source == "programmatic_decision_v1" and row["task"].startswith("binding:")
    ):
        field = row.get("field") or row.get("rule", {}).get("field") or row["task"]
        value = str(state[field])
        valid = {f"{field}: {value}", f"The {field} is {value}."}
        matches = [opt for opt in row["options"] if opt in valid]
        if len(matches) != 1:
            raise ValueError("binding target absent or ambiguous")
        return matches[0]
    if source == "programmatic_decision_v1":
        rule = row["rule"]
    elif source == "curriculum":
        q = row["question"]
        match = re.fullmatch(
            r"Select the item with the (minimum|maximum) (cost|risk|time) value\.", q
        )
        if match:
            rule = dict(operation="extremum", direction=match[1], field=match[2])
        elif match := re.fullmatch(r"How many items have risk at most (\d+)\?", q):
            rule = dict(operation="count", threshold=int(match[1]))
        elif match := re.fullmatch(
            r"Among items with risk at most (\d+), select the item with the minimum cost\. "
            r"Select none if no item qualifies\.", q
        ):
            rule = dict(operation="filter_min_cost", threshold=int(match[1]))
        else:
            raise ValueError("unknown curriculum wording; do not guess")
    else:
        raise ValueError("unsupported text source")
    if rule["operation"] == "extremum":
        values = [s[rule["field"]] for s in state]
        if rule["direction"] not in {"minimum", "maximum"}:
            raise ValueError("unknown comparison direction")
        extreme = (min if rule["direction"] == "minimum" else max)(values)
        winners = [s["id"] for s in state if s[rule["field"]] == extreme]
        if len(winners) != 1:
            raise ValueError("ambiguous numeric tie")
        answer = winners[0]
    else:
        eligible = [s for s in state if s["risk"] <= rule["threshold"]]
        if rule["operation"] == "count":
            answer = str(len(eligible))
        elif rule["operation"] == "filter_min_cost":
            if eligible:
                best = min(s["cost"] for s in eligible)
                winners = [s["id"] for s in eligible if s["cost"] == best]
                if len(winners) != 1:
                    raise ValueError("ambiguous filtered tie")
                answer = winners[0]
            else:
                answer = "none"
        else:
            raise ValueError("unknown numeric operation")
    if source == "programmatic_decision_v1":
        return row["options"][row["option_ids"].index(answer)]
    return answer


def label_coverage(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"] + ":" + row["task"]].append(row)
    result = {}
    for key, group in sorted(groups.items()):
        counts = Counter(r["options"][r["target_index"]] for r in group)
        candidates = set(o for r in group for o in r["options"])
        item = dict(rows=len(group), unique_rows=len({r["id"] for r in group}),
                    observations=len({r["observation"] for r in group}),
                    groups=len({r["group"] for r in group}), targets=dict(sorted(counts.items())),
                    candidates_without_target=sorted(candidates - counts.keys()),
                    majority_fraction=max(counts.values()) / len(group))
        if group[0]["source"] == "starss":
            item.update(positive=counts["yes"], negative=counts["no"],
                        always_no_accuracy=counts["no"] / len(group))
        result[key] = item
    return result


def annotation_counts(csv_text, start_seconds, end_seconds):
    """Independent STARSS official 100ms, half-open interval label calculation."""
    import csv
    import io

    if not 0 <= start_seconds < end_seconds:
        raise ValueError("invalid annotation interval")
    frames = [set() for _ in range(12)]
    for fields in csv.reader(io.StringIO(csv_text)):
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError("six official annotation columns required")
        frame, label, _, _, _, _ = map(int, fields)
        if frame < 0 or not 0 <= label <= 12:
            raise ValueError("invalid official class/frame")
        if start_seconds * 10 <= frame < end_seconds * 10:
            frames[max(0, label - 1)].add(frame)
    return [len(f) for f in frames]


def crop_geometry(width, height, size=288):
    if any(type(v) is not int or v < 1 for v in (width, height, size)):
        raise ValueError("positive integer dimensions required")
    scale = max(size / height, size / width)
    nh, nw = round(height * scale), round(width * scale)
    left, top = (nw - size) // 2, (nh - size) // 2
    return dict(original=[width, height], resized=[nw, nh],
                retained_fraction=size * size / (nw * nh),
                source_box=[left * width / nw, top * height / nh,
                            (left + size) * width / nw, (top + size) * height / nh])


def independent_feature_slice(array, spec):
    """Explicit indexing independently mirrors the transport's cached slicing."""
    x = array
    if "index" in spec:
        if type(spec["index"]) is not int or not 0 <= spec["index"] < len(x):
            raise ValueError("invalid observation index")
        x = x[spec["index"]]
    start, count = spec["offset"], spec["tokens"]
    if (type(start) is not int or type(count) is not int or start < 0 or count < 1
            or x.ndim != 2 or start + count > len(x)):
        raise ValueError("invalid observation slice")
    value = x[np.arange(start, start + count)]
    width = 768 if spec["kind"] in {"image", "video"} else 384
    if value.shape != (count, width) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("invalid feature values")
    return value


def independent_metrics(logits, targets):
    """Independent probability/15-bin top-label calibration recomputation."""
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    if (logits.ndim != 2 or not len(logits) or not np.isfinite(logits).all()
            or targets.shape != (len(logits),) or targets.dtype.kind not in "iu"
            or (targets < 0).any() or (targets >= logits.shape[1]).any()):
        raise ValueError("finite logits and aligned valid integer targets required")
    relative = logits - np.max(logits, axis=1)[:, None]
    log_z = np.log(np.sum(np.exp(relative), axis=1))
    probability = np.exp(relative - log_z[:, None])
    correct = logits.argmax(1) == targets
    confidence = probability.max(1)
    squared = np.square(probability).sum(1) + 1 - 2 * probability[np.arange(len(targets)), targets]
    ece = 0.0
    for b in range(15):
        selected = (confidence >= b / 15) & (
            confidence < (b + 1) / 15 if b < 14 else confidence <= 1)
        if selected.any():
            ece += abs(float(correct[selected].sum() - confidence[selected].sum())) / len(targets)
    return dict(count=len(targets), accuracy=float(correct.mean()),
                nll=float((log_z - relative[np.arange(len(targets)), targets]).mean()),
                brier=float(squared.mean()), ece=ece, mean_confidence=float(confidence.mean()))
