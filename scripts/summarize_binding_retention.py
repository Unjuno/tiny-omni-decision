"""Verify and summarize the fixed parent-KL fit; no model or transfer loading."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import select_checkpoint
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.summarize_condition_binding import gate_failures, score_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--primary-sha256", required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = args.root / "condition-binding-retention-fit-01"
    replay = args.root / "condition-binding-retention-replay-01"
    control = args.root / "condition-binding-fit-01"
    corpus = args.root / "condition-binding-corpus-01"
    verify(
        {
            str(primary / "result.json"): args.primary_sha256,
            str(replay / "result.json"): args.replay_sha256,
        }
    )
    fit, repeat, old = (read(p / "result.json") for p in (primary, replay, control))
    if fit["status"] != "PASS_FIXED_BINDING_RETENTION" or fit["steps"] != 512:
        raise ValueError("primary incomplete")
    if repeat["status"] != "PASS_SELECTED_RETENTION_REPLAY" or repeat["steps"] != 0:
        raise ValueError("fresh selected inference replay incomplete")
    for directory, result in ((primary, fit), (replay, repeat)):
        verify({str(directory / p): h for p, h in result["artifacts"].items()})
        verify(result["input_pins"])
        verify(result["code_pins"])
        if result["transfer_evaluated"] or result["sealed_audit_loaded"]:
            raise ValueError("unplanned evaluation")
    if fit["selected_sha256"] != repeat["selected_sha256"]:
        raise ValueError("selected weights changed")
    history, previous = (read(p / "history.json") for p in (primary, control))
    if [r["step"] for r in history] != [0, 128, 256, 384, 512]:
        raise ValueError("incomplete curve")
    if select_checkpoint(history) != fit["best_step"]:
        raise ValueError("selector mismatch")
    curves = []
    for current, baseline in zip(history, previous, strict=True):
        failures = gate_failures(current, history[0])
        if bool(failures) == current["eligible"] or current["step"] != baseline["step"]:
            raise ValueError("gate or step mismatch")
        curves.append(dict(candidate=current, control=baseline, gate_failures=failures))
    updates = read(primary / "updates.json")
    if [r["step"] for r in updates] != list(range(1, 513)):
        raise ValueError("incomplete update accounting")
    cache = read(primary / "teacher-cache.json")
    if cache["summary"]["questions"] != 2048 or len(cache["entries"]) != 2048:
        raise ValueError("teacher cache coverage mismatch")
    selected = next(r for r in history if r["step"] == fit["best_step"])
    intervals = {}
    for label, step, ref_step in (
        ("selected_vs_parent", fit["best_step"], 0),
        ("final_vs_parent", 512, 0),
        ("final_vs_control_final", 512, 512),
    ):
        intervals[label] = {}
        for source, file in (
            ("binding", "validation"),
            ("original", "original"),
            ("curriculum", "curriculum"),
        ):
            data = lines(corpus / f"{file}.jsonl")
            before = np.load(
                control / f"validation-{ref_step}-{source}-scores.npy", allow_pickle=False
            )
            after = np.load(primary / f"validation-{step}-{source}-scores.npy", allow_pickle=False)
            interval = score_difference(data, before, after)
            interval["columns"] = [
                "accuracy_candidate_minus_reference",
                "nll_candidate_minus_reference",
            ]
            intervals[label][source] = interval
    environment = read(primary / "effective-config.json")
    result = dict(
        status="PASS_COMPLETED_FIXED_RETENTION_COMPARISON",
        product_goal_met=False,
        new_checkpoint_retained=fit["best_step"] > 0,
        primary=fit,
        fresh_selected_replay=repeat,
        control=dict(
            path=str(control),
            sha256=sha(control / "result.json"),
            source_commit=old["source_commit"],
            selected_step=old["best_step"],
        ),
        environment=environment["environment"],
        configuration=environment["config"],
        input_corpus_sha256=sha(corpus / "result.json"),
        hashes={
            str(primary / "result.json"): args.primary_sha256,
            str(replay / "result.json"): args.replay_sha256,
        },
        curves=curves,
        paired_intervals=intervals,
        full_consumed_questions=history[-1]["consumed"],
        selected_consumed_questions=selected["consumed"],
        optimization_seconds=sum(r["seconds"] for r in updates),
        scheduled_validation_seconds=sum(r["evaluation_seconds"] for r in history),
        initial_kl=updates[0]["retention_kl"],
        kl_range=[min(r["retention_kl"] for r in updates), max(r["retention_kl"] for r in updates)],
        gradient_norm_range=[
            min(r["gradient_norm"] for r in updates),
            max(r["gradient_norm"] for r in updates),
        ],
        teacher_cache=cache["summary"],
        scope="Single-seed observed development comparison; selection bias, no multiplicity "
        "correction, no transfer evaluation or native-modality quality claim.",
        summarizer_sha256=sha(__file__),
    )
    if args.output.exists():
        raise ValueError("refusing to overwrite publication")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(
        json.dumps(
            dict(output=str(args.output), sha256=sha(args.output), best_step=fit["best_step"])
        )
    )


if __name__ == "__main__":
    main()
