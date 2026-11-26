"""Freeze the one-factor native support experiment using TRAIN labels only."""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.native_support_sampling import POLICY, repair_support
from scripts.native_video_cache import sha

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    args = parser.parse_args()
    if args.config_output.exists():
        raise FileExistsError(args.config_output)
    baseline_config = REPO / "configs/pretrained_reuse/native_long_budget_v1.json"
    if sha(baseline_config) != "19e4a1f005856684950fb229a12d757768eb7369787ad08b40b3f85de2ba3e45":
        raise ValueError("original long-run config differs")
    config = json.loads(baseline_config.read_text(encoding="utf-8"))
    fit = ROOT / "native-long-budget-fit-01"
    if sha(fit / "result.json") != (
        "645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b"
    ):
        raise ValueError("completed baseline receipt differs")
    baseline = json.loads((fit / "result.json").read_text())
    if sha(fit / "sampling.json") != baseline["artifacts"]["sampling.json"]:
        raise ValueError("baseline sampler differs")
    corpus = Path(config["corpus"])
    receipt = json.loads((corpus / "result.json").read_text())
    if sha(corpus / "train.jsonl") != receipt["artifacts"]["train.jsonl"]:
        raise ValueError("frozen TRAIN differs")
    train = [json.loads(line) for line in (corpus / "train.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    previous = json.loads((fit / "sampling.json").read_text())
    plan = repair_support(train, previous["order"])
    if (len(plan["order"]) != 32768 or len(plan["unique_row_indices"]) != 6656
            or len(plan["events"]) != 12):
        raise ValueError("original budget/counts required")
    parent = {train[i]["id"] for i in plan["parent_unique_row_indices"]}
    current = {train[i]["id"] for i in plan["unique_row_indices"]}
    union = dict(parent_ids=sorted(parent), additional_ids=sorted(current - parent),
                 union_ids=sorted(parent | current), consumed_ids=sorted(current))
    args.output.mkdir(exist_ok=False)
    write(args.output / "plan.json", plan)
    write(args.output / "union-train-ids.json", union)
    config.update(id="native-starss-support-floor-v1", policy=POLICY,
                  parent_fit=str(fit), parent_sampling=str(fit / "sampling.json"),
                  support_plan=str(args.output / "plan.json"),
                  union_probe=str(args.output / "union-train-ids.json"),
                  training_probe_scope="All6656 unique questions consumed by support policy; "
                      "different STARSS membership from parent; full validation unchanged.",
                  interpretation="Only within-STARSS-event negative-to-positive TRAIN row "
                      "substitution. Preserve all existing positives, non-STARSS slots, "
                      "per-event quota, unique count and repetition multiplicity. "
                      "New seed17 initialization and optimizer; same32768 LR horizon. "
                      "One predefined availability-capped policy, not an optimized ratio.")
    del config["pools"], config["plans"]
    config["input_pins"] = {p: h for p, h in config["input_pins"].items()
                            if "native-coverage-plan-02" not in p
                            or p == config["state_plan"]}
    for path in (baseline_config, fit / "result.json", fit / "sampling.json",
                 args.output / "plan.json", args.output / "union-train-ids.json"):
        config["input_pins"][str(path)] = sha(path)
    write(args.config_output, config)
    report = dict(status="PASS_FROZEN_SUPPORT_PLAN", policy=POLICY,
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  source_bytes={name: sha(REPO / "scripts" / name) for name in (
                      "prepare_native_support.py", "native_support_sampling.py")},
                  config_path=str(args.config_output), config_sha256=sha(args.config_output),
                  input_hashes={str(corpus / "train.jsonl"): sha(corpus / "train.jsonl"),
                                str(fit / "sampling.json"): sha(fit / "sampling.json")},
                  artifacts={name: sha(args.output / name) for name in (
                      "plan.json", "union-train-ids.json")},
                  replacements=len(plan["replacements"]), events=plan["events"],
                  validation_loaded=False, calibration_loaded=False, sealed_audit_loaded=False,
                  optimizer_updates=0)
    write(args.output / "result.json", report)
    print(json.dumps(dict(replacements=report["replacements"], events=report["events"]), indent=2))


if __name__ == "__main__":
    main()
