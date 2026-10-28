"""Independently replay and summarize the fixed GRID AV matching readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.fit_grid_path_a_av_matching import (
    EVAL_SEED,
    AVMatchReadout,
    listwise_metrics,
    load_groups,
    make_arrays,
    normalize,
)
from scripts.fit_grid_path_a_readouts import COHORT, FEATURES, sha, validate_inputs


def clustered_bootstrap_delta(
    matched: np.ndarray,
    control: np.ndarray,
    speakers: np.ndarray,
    seed: int = 1703,
    replicates: int = 10_000,
) -> dict:
    identities = list(dict.fromkeys(speakers.tolist()))
    per_speaker = {
        identity: float((matched[speakers == identity] - control[speakers == identity]).mean())
        for identity in identities
    }
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(identities, size=len(identities), replace=True)
        values[index] = np.mean([per_speaker[item] for item in sampled])
    return {
        "estimand": "unweighted mean of speaker-level matched-minus-control accuracy",
        "speaker_macro_delta": float(np.mean(list(per_speaker.values()))),
        "percentile_95_interval": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "speaker_count": len(identities),
        "replicates": replicates,
        "seed": seed,
        "per_speaker_delta": per_speaker,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "effective-config.json").read_text(encoding="utf-8"))
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    archive = np.load(run_dir / "joint-validation-controls.npz", allow_pickle=False)

    audio, video, _, ids, cohort_from_features, _ = validate_inputs()
    if cohort != cohort_from_features:
        raise ValueError("cohort changed since run")
    groups = load_groups(cohort, ids)
    train = make_arrays(groups["train"], audio, ids)
    validation = make_arrays(groups["validation"], audio, ids)
    train_video = video[train["query_rows"]]
    validation_video = video[validation["query_rows"]]
    (
        train["audio"],
        validation["audio"],
        _,
        _,
        _,
        _,
        video_mean,
        video_scale,
    ) = normalize(train["audio"], validation["audio"], train_video, validation_video)
    validation["video_bank"] = ((video - video_mean) / video_scale).astype(np.float32)
    rng = np.random.default_rng(EVAL_SEED)
    orders = np.stack([rng.permutation(4) for _ in validation["targets"]])
    validation["ordered_candidate_rows"] = np.take_along_axis(
        validation["candidate_rows"], orders, 1
    )
    validation["ordered_candidate_ids"] = np.take_along_axis(validation["candidate_ids"], orders, 1)
    validation["ordered_targets"] = np.argmax(orders == validation["targets"][:, None], axis=1)
    speakers = np.asarray([identity.rsplit("_", 1)[0] for identity in validation["query_ids"]])
    if len(validation["targets"]) != 256 or len(set(speakers)) != 8:
        raise ValueError("fixed validation cohort mismatch")
    if set(config["train_speakers"]) & set(config["validation_speakers"]):
        raise ValueError("speaker leakage")
    if not np.array_equal(validation["query_ids"], archive["source_ids"].astype(str)):
        raise ValueError("validation source order changed")
    if not np.array_equal(validation["ordered_targets"], archive["targets"]):
        raise ValueError("validation target order changed")
    if not np.array_equal(
        validation["ordered_candidate_ids"], archive["candidate_ids"].astype(str)
    ):
        raise ValueError("candidate option order changed")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    target_tensor = torch.from_numpy(validation["ordered_targets"].astype(np.int64))
    candidates = torch.from_numpy(
        np.asarray(validation["video_bank"][validation["ordered_candidate_rows"]], dtype=np.float32)
    )
    replay = {}
    for name, mask in (("audio", (1.0, 0.0)), ("video", (0.0, 1.0)), ("joint", (1.0, 1.0))):
        checkpoint = run_dir / f"{name}-best.pt"
        if sha(checkpoint) != result[name]["best_state_sha256"]:
            raise ValueError(f"{name} selected checkpoint hash mismatch")
        model = AVMatchReadout().cpu()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(validation["audio"]),
                candidates,
                torch.tensor(mask).repeat(256, 1),
            )
        metrics = listwise_metrics(logits, target_tensor)
        saved = np.load(run_dir / f"{name}-validation-predictions.npz", allow_pickle=False)
        probability_delta = np.max(np.abs(metrics["probabilities"] - saved["probabilities"]))
        argmax_changes = int(np.count_nonzero(metrics["predictions"] != saved["predictions"]))
        if argmax_changes:
            raise ValueError(f"{name} reloaded checkpoint prediction mismatch")
        if metrics["accuracy"] != result[name]["selected"]["accuracy"]:
            raise ValueError(f"{name} aggregate Accuracy mismatch")
        replay[name] = {
            "selected_epoch": result[name]["best_epoch"],
            "selected_metrics": {
                key: value
                for key, value in metrics.items()
                if key not in {"probabilities", "predictions"}
            },
            "max_probability_delta": float(probability_delta),
            "argmax_changes": argmax_changes,
            "checkpoint_sha256": sha(checkpoint),
        }

    matched_correct = (archive["matched_probabilities"].argmax(1) == archive["targets"]).astype(
        np.float64
    )
    paired = {}
    for control in (
        "audio_missing",
        "video_missing",
        "both_missing",
        "audio_shuffled_within_quartet",
        "stale_video_other_quartet_same_speaker",
        "video_temporal_offset_4",
    ):
        control_correct = (
            archive[f"{control}_probabilities"].argmax(1) == archive["targets"]
        ).astype(np.float64)
        paired[control] = clustered_bootstrap_delta(matched_correct, control_correct, speakers)

    stale_ids = archive["stale_video_candidate_ids"].astype(str)
    source_ids = archive["source_ids"].astype(str)
    if any(query in choices for query, choices in zip(source_ids, stale_ids, strict=True)):
        raise ValueError("stale control includes the correct source utterance")
    output = {
        "status": "PASS_GRID_PATH_A_AV_MATCHING_REPLAY",
        "run_result_sha256": sha(run_dir / "result.json"),
        "effective_config_sha256": sha(run_dir / "effective-config.json"),
        "cohort_sha256": sha(COHORT),
        "feature_result_sha256": sha(FEATURES / "result.json"),
        "speaker_disjoint": True,
        "validation_examples": 256,
        "validation_speakers": 8,
        "selected_checkpoint_replay": replay,
        "controls": result["joint_controls"],
        "matched_minus_control_accuracy_bootstrap": paired,
        "bootstrap_warning": (
            "Eight validation speakers only; repeated development validation and "
            "best-checkpoint selection "
            "make intervals descriptive, not blind confirmatory inference."
        ),
    }
    output_path = run_dir / "analysis.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "analysis_sha256": sha(output_path),
                "replay_max_probability_delta": {
                    key: value["max_probability_delta"] for key, value in replay.items()
                },
                "argmax_changes": {key: value["argmax_changes"] for key, value in replay.items()},
                "bootstrap_intervals": {
                    key: value["percentile_95_interval"] for key, value in paired.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
