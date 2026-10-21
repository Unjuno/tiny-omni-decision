"""Describe saved multi-option ranks; this does not evaluate a yes/no model."""

import hashlib
import json
import math
from pathlib import Path


def summarize(rows):
    groups = {}
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate question")
        seen.add(row["id"])
        options, p = row["options"], row["probabilities"]
        if len(options) < 2 or len(set(options)) != len(options) or len(p) != len(options):
            raise ValueError("invalid options")
        if any(not math.isfinite(x) or x <= 0 for x in p) or abs(sum(p) - 1) > 1e-5:
            raise ValueError("invalid probabilities")
        gold = options.index(row["target"])
        pairs = [p[gold] / (p[gold] + x) for i, x in enumerate(p) if i != gold]
        g = groups.setdefault(len(p), [])
        g.append({
            "accuracy": float(max(range(len(p)), key=p.__getitem__) == gold),
            "strict_all_wins": float(all(x > .5 for x in pairs)),
            "pair_win_rate_ties_half": sum(1 if x > .5 else .5 if x == .5 else 0
                                           for x in pairs) / len(pairs),
            "pair_nll": sum(-math.log(x) for x in pairs) / len(pairs),
            "original_nll": -math.log(p[gold]),
        })
    return {
        str(n): {"questions": len(values), "gold_distractor_pairs": len(values) * (n - 1),
                 **{key: sum(v[key] for v in values) / len(values) for key in values[0]}}
        for n, values in sorted(groups.items())
    }


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/text-coverage-training-01")
    report = {"scope": "Observed development predictions; no inference or selection",
              "checkpoint_step": 1792, "sources": {}}
    for source in ("original", "curriculum"):
        path = root / f"predictions-1792-{source}.json"
        raw = path.read_bytes()
        report["sources"][source] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "by_option_count": summarize(json.loads(raw)),
        }
    output = Path("docs/pretrained_reuse/R2_OPTION_CARDINALITY_20261012")
    if output.with_suffix(".json").exists():
        raise FileExistsError("preserve existing report")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Candidate cardinality analysis", "",
             "Fixed selected checkpoint 1792; saved development predictions only.",
             "No model inference, training, calibration or checkpoint selection.", "",
             "| Source | Options | Questions | Original Accuracy | Pair win rate |"
             " Original NLL | Pair NLL |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for source, data in report["sources"].items():
        for n, g in data["by_option_count"].items():
            lines.append(f"| {source} | {n} | {g['questions']} | {g['accuracy']:.6f} | "
                         f"{g['pair_win_rate_ties_half']:.6f} | {g['original_nll']:.6f} | "
                         f"{g['pair_nll']:.6f} |")
    lines += ["", "## Interpretation and limitations", "",
              "Each pair contains the known gold option and one distractor. All distractors",
              "are included, with each question equally weighted within its option-count group.",
              "Pair probability is p(gold)/(p(gold)+p(distractor)); ties score one half.",
              "Original Accuracy uses the saved option order for argmax ties.",
              "This uses gold information to construct pairs: it is not an inference policy",
              "available before knowing the answer, nor a yes/no entailment evaluation.",
              "Removing distractors changes the task and chance level (1/K versus 1/2).",
              "Higher pair scores cannot count toward the original multi-option 95% gate.",
              "Questions with different option counts also differ in content; do not infer",
              "a causal option-count effect from cross-group Accuracy comparisons.",
              "Pair NLL has a different sample space and cannot establish better calibration.",
              "Pairs share questions/states; no independent-pair confidence claim is made.",
              "", "Keep a variable-option product contract, including two-option questions.",
              "A learned yes/no reformulation would require separately frozen, semantically",
              "equivalent questions and joint consistency evaluation. This analysis does not",
              "prove which trained architecture is superior. No product model is promoted."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
