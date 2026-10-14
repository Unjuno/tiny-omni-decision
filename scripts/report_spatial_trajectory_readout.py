"""Audit all saved epochs and publish a small exact paired-readout result."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.audit_correspondence_validation import sha, write
from scripts.cover_point_metrics import membership_metrics
from scripts.train_role_supervised_qa import load, qa_summary
from scripts.train_spatial_trajectory_readout import paired_video_bootstrap


def verify_original_labels(saved, original):
    if set(saved) != set(original) or any(
        saved[key].dtype != original[key].dtype or not np.array_equal(saved[key], original[key])
        for key in original
    ):
        raise ValueError("saved labels/identities differ from original pinned corpus")


def verify_metrics(computed, recorded):
    if isinstance(computed, dict):
        if not isinstance(recorded, dict) or set(computed) != set(recorded):
            raise ValueError("saved metric groups mismatch")
        for key, value in computed.items():
            verify_metrics(value, recorded[key])
    elif computed is None:
        if recorded is not None:
            raise ValueError("saved metrics mismatch")
    elif isinstance(computed, list):
        if computed != recorded:
            raise ValueError("saved metrics mismatch")
    elif not np.isclose(computed, recorded, rtol=0, atol=1e-12):
        raise ValueError("saved original QA metrics mismatch")


def audit_epoch(saved, labels, recorded, auxiliary=None):
    for split, truth in labels.items():
        p, logp = saved[split], saved[split + "_logp"]
        if p.shape != (len(truth), 3) or logp.shape != p.shape:
            raise ValueError("original QA denominator changed")
        if not np.isfinite(p).all() or not np.isfinite(logp).all():
            raise ValueError("nonfinite saved QA outputs")
        if not np.allclose(p, np.exp(logp), rtol=1e-5, atol=1e-7):
            raise ValueError("saved probability/log probability inconsistency")
        computed = qa_summary(p, truth, logp)
        verify_metrics(computed, recorded[split])
        if auxiliary is not None:
            cover, available = auxiliary[split]
            verify_metrics(
                membership_metrics(saved[split + "_cover"][available], cover[available]),
                recorded[split + "_cover_supplemental"],
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.curves.exists():
        raise FileExistsError("publication outputs already exist")
    result_path, config_path = args.run / "result.json", args.run / "config.json"
    result, config = json.loads(result_path.read_text()), json.loads(config_path.read_text())
    if result["status"] != "PASS_FIXED_SPATIAL_APPEARANCE_COMPARISON_ONLY" or result["smoke"]:
        raise ValueError("complete real comparison required")
    if sha(config_path) != result["config_sha256"]:
        raise ValueError("effective config changed")
    expected_arms = {"initial_broadcast", "temporal_appearance"}
    if set(result["arms"]) != expected_arms:
        raise ValueError("exactly two paired arms required")
    for path, pin in config["input_hashes"].items():
        if sha(Path(path)) != pin:
            raise ValueError("original protected input changed")
    bank, original, _, _ = load()
    del bank
    labels, auxiliary, label_hashes = {}, {}, {}
    for split, count in (("train", 198), ("valid", 514)):
        label_path = args.run / f"{split}-labels.npz"
        with np.load(label_path, allow_pickle=False) as saved:
            d = original[split]
            verify_original_labels(
                saved,
                dict(
                    qa=d["targets"].numpy(),
                    cover=d["cover_targets"].numpy(),
                    available=d["cover_available"].numpy(),
                    identities=np.array([p["video_id"] for p in d["pairs"]]),
                ),
            )
            labels[split] = saved["qa"].copy()
            auxiliary[split] = (saved["cover"].copy(), saved["available"].copy())
            if len(labels[split]) != count or len(saved["identities"]) != count // 2:
                raise ValueError("original QA/video count changed")
        label_hashes[split] = sha(label_path)
    curves, publication, checked = [], {}, 0
    for arm, metadata in result["arms"].items():
        folder = args.run / arm
        for name, pin in metadata["hashes"].items():
            if sha(folder / name) != pin:
                raise ValueError("saved artifact hash mismatch")
            checked += 1
        history = json.loads((folder / "history.json").read_text())
        orders = json.loads((folder / "orders.json").read_text())
        if len(history) != 50 or len(orders) != 50 or metadata["updates"] != 650:
            raise ValueError("fixed full budget incomplete")
        for epoch, (row, order) in enumerate(zip(history, orders, strict=True), 1):
            if (
                row["epoch"] != epoch
                or row["updates"] != 13 * epoch
                or sorted(order) != list(range(99))
            ):
                raise ValueError("epoch/update/sampling accounting changed")
            with np.load(
                folder / f"epoch-{epoch:02d}-predictions.npz", allow_pickle=False
            ) as saved:
                audit_epoch(saved, labels, row, auxiliary)
                for split, count in (("train", 99), ("valid", 257)):
                    if saved[split + "_cover"].shape != (count, 256):
                        raise ValueError("cover output denominator changed")
            curves.append(
                dict(
                    arm=arm,
                    epoch=epoch,
                    updates=row["updates"],
                    train_accuracy=row["train"]["accuracy"],
                    train_nll=row["train"]["nll"],
                    validation_accuracy=row["valid"]["accuracy"],
                    validation_nll=row["valid"]["nll"],
                    validation_brier=row["valid"]["brier"],
                    validation_ece=row["valid"]["ece"],
                    validation_both_correct=row["valid"]["both_correct"],
                    seconds=row["seconds"],
                    best_epoch=row["best_epoch"],
                )
            )
        chosen = int(np.argmin([row["valid"]["nll"] for row in history])) + 1
        if chosen != metadata["best_epoch"]:
            raise ValueError("earliest minimum original QA NLL selector mismatch")
        if metadata["run_video_presentations"] != 4950 or metadata["unique_train_videos"] != 99:
            raise ValueError("run/unique accounting mismatch")
        if metadata["selected_video_presentations"] != chosen * 99:
            raise ValueError("selected checkpoint accounting mismatch")
        publication[arm] = dict(
            metadata,
            selected=history[chosen - 1],
            final=history[-1],
            unique_train_questions=198,
            repeated_video_presentations=4851,
            repeated_question_presentations=9702,
        )
    a, b = list(publication.values())
    if a["initial_sha256"] != b["initial_sha256"] or a["orders_sha256"] != b["orders_sha256"]:
        raise ValueError("paired initialization/order mismatch")
    with (
        np.load(args.run / "initial_broadcast/selected-controls.npz") as a,
        np.load(args.run / "temporal_appearance/selected-controls.npz") as b,
    ):
        bootstrap = paired_video_bootstrap(
            a["normal"], a["normal_logp"], b["normal"], b["normal_logp"], labels["valid"]
        )
        if bootstrap != result["paired_video_bootstrap"]:
            raise ValueError("paired-bootstrap replay mismatch")
        selected_diagnostics = {}
        truth = labels["valid"].reshape(-1, 2)
        changed = truth[:, 0] != truth[:, 1]
        for arm, saved in (("initial_broadcast", a), ("temporal_appearance", b)):
            best_epoch = publication[arm]["best_epoch"]
            with np.load(args.run / arm / f"epoch-{best_epoch:02d}-predictions.npz") as selected:
                for suffix in ("", "_cover", "_logp"):
                    np.testing.assert_array_equal(
                        saved["normal" + suffix], selected["valid" + suffix]
                    )
            for control in publication[arm]["controls"]:
                audit_epoch(
                    dict(valid=saved[control], valid_logp=saved[control + "_logp"]),
                    dict(valid=labels["valid"]),
                    dict(valid=publication[arm]["controls"][control]),
                )
            predicted = saved["normal"].argmax(-1).reshape(-1, 2)
            selected_diagnostics[arm] = dict(
                changed_target_videos=int(changed.sum()),
                changed_target_both_correct=int(
                    (predicted[changed] == truth[changed]).all(1).sum()
                ),
                different_condition_predictions=int((predicted[:, 0] != predicted[:, 1]).sum()),
                argmax_counts=np.bincount(predicted.ravel(), minlength=3).tolist(),
                control_answer_changes={
                    name: int((saved[name].argmax(-1) != predicted.ravel()).sum())
                    for name in ("zero", "mismatch", "initial", "reverse")
                },
            )
    with args.curves.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    output = dict(
        status="PASS_ALL100_EPOCH_METRICS_AND_HASHES_AUDITED_NOT_OMNI_QUALITY",
        source_run=str(args.run),
        source_commit=config["source_commit"],
        config_sha256=sha(config_path),
        result_sha256=sha(result_path),
        curves_sha256=sha(args.curves),
        artifacts_checked=checked,
        original_bound_label_hashes=label_hashes,
        protocol="R2_SPATIAL_READOUT_PROTOCOL_20261014.md",
        arms=publication,
        selected_diagnostics=selected_diagnostics,
        paired_video_bootstrap=bootstrap,
        seconds=result["seconds"],
        started_utc=result["started_utc"],
        ended_utc=result["ended_utc"],
        cloud_cost=0,
        peak_allocated_bytes=result["peak_allocated_bytes"],
        peak_reserved_bytes=result["peak_reserved_bytes"],
        peak_process_working_set_bytes=result["peak_process_working_set_bytes"],
        total_video_text_readout_parameters=31072327 + 17389824 + 232385,
        full_four_modality_model=False,
        sealed_audit_opened=False,
    )
    write(args.output, output)
    print(output["status"], checked, "artifacts", flush=True)


if __name__ == "__main__":
    main()
