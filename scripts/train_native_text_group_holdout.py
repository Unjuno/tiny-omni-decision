"""Fit one text-only frozen-feature readout against a TRAIN-only group holdout."""

# Establish deterministic CUDA workspace and offline execution before importing torch.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from copy import copy
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

from scripts.native_candidate_inputs import positions
from scripts.native_fit import learning_rate, option_order, summarize
from scripts.native_text_state_holdout import (
    SOURCE as CONDITION_RELEVANCE_SOURCE,
)
from scripts.native_text_state_holdout import (
    SPLIT_ID as CONDITION_RELEVANCE_SPLIT_ID,
)
from scripts.native_text_state_holdout import (
    balanced_order,
    build_split,
    counterfactual_donor_indices,
)
from scripts.native_video_cache import sha
from tiny_omni_decision.native_candidate import NativeCandidateReadout, NativeTokenBlock

DEFAULT_CONFIG = REPO / "configs/pretrained_reuse/native_text_state_group_holdout_v1.json"
CHECKPOINT_INTERVAL = 256
UPDATES = 2048
BOOTSTRAP_REPLICATES = 5000
SPLIT_PROFILES = {
    CONDITION_RELEVANCE_SOURCE: {
        "split_id": CONDITION_RELEVANCE_SPLIT_ID,
        "eligible_groups": 130,
        "heldout_groups": 65,
    },
    "text:programmatic_decision_v1": {
        "split_id": "native-text-programmatic-group-holdout-v1",
        "eligible_groups": 513,
        "heldout_groups": 256,
    },
}
COUNTERFACTUAL_INTERVENTION = "same-task counterfactual state with mapped donor target"


def configured_source_split(config):
    source = config.get("source")
    if source is None and config.get("id") == "native-text-state-group-holdout-v1":
        source = CONDITION_RELEVANCE_SOURCE
    profile = SPLIT_PROFILES.get(source)
    split_id = config.get("split_id", profile["split_id"] if profile else None)
    return source, split_id


def is_local_c_absolute_path(value):
    """Validate the configured Windows artifact volume independent of host OS.

    The experiment configs intentionally describe a Windows C: artifact root.
    ``Path`` on POSIX treats that string as relative, so config-only CI checks
    must parse it using Windows path rules. Actual writes still pass through
    ``validate_output_path`` and the platform-native ``Path`` checks.
    """
    path = PureWindowsPath(value)
    return path.is_absolute() and path.drive.upper() == "C:"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_pins(pins):
    for path, expected in pins.items():
        observed = sha(Path(path))
        if observed != expected:
            raise ValueError(f"pinned input changed: {path} ({observed})")


def validate_config(config):
    expected_training = {
        "seed": 17,
        "updates": 2048,
        "microbatches_per_update": 4,
        "optimizer": "AdamW",
        "learning_rate_peak": 0.001,
        "warmup_updates": 62,
        "scheduler": "cosine to zero at update 2048",
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_policy": "fixed terminal checkpoint; no holdout checkpoint selection",
    }
    if config.get("training") != expected_training:
        raise ValueError("training config differs from the fixed experiment implementation")
    source, split_id = configured_source_split(config)
    profile = SPLIT_PROFILES.get(source)
    if not profile or split_id != profile["split_id"]:
        raise ValueError("unsupported source/split identity")
    split = config.get("split", {})
    if (
        split.get("seed") != 17
        or split.get("holdout_fraction_of_eligible_groups") != 0.5
        or split.get("eligible_groups_not_seen_by_prior_2048_step_sampler")
        != profile["eligible_groups"]
        or split.get("heldout_group_count") != profile["heldout_groups"]
    ):
        raise ValueError("group split differs from the preregistered TRAIN-only source profile")
    evaluation = config.get("evaluation", {})
    if (
        evaluation.get("official_validation_loaded") is not False
        or evaluation.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("official validation and sealed audit must remain closed")
    interventions = evaluation.get("primary_interventions")
    if not isinstance(interventions, list) or interventions[:2] != [
        "same input",
        "zero text state",
    ]:
        raise ValueError("evaluation must include fixed original and zero-state arms")
    if any(
        value
        not in {
            "same input",
            "zero text state",
            COUNTERFACTUAL_INTERVENTION,
        }
        for value in interventions
    ):
        raise ValueError("unsupported input intervention")
    if len(interventions) != len(set(interventions)):
        raise ValueError("duplicate input intervention")
    if not is_local_c_absolute_path(config["output_root"]):
        raise ValueError("artifact output root must resolve to the local C: volume")
    if config.get("staging_receipt_sha256") == "TO_BE_FILLED_AFTER_LOCAL_COPY":
        raise ValueError("local frozen-input copy must be hashed and pinned first")
    if config["input_pins"].get(config["staging_receipt_path"]) != config["staging_receipt_sha256"]:
        raise ValueError("staging receipt must be included in the pinned input inventory")


def validate_output_path(config, output):
    root = Path(config["output_root"]).resolve()
    destination = Path(output).resolve()
    if destination.drive.upper() != "C:" or not destination.is_relative_to(root):
        raise ValueError("output must be a new child of the configured local C: artifact root")


def validate_split_membership(fit_rows, holdout_rows, previously_seen_groups, source):
    rows = fit_rows + holdout_rows
    if not rows or any(row.get("source") != source or row.get("split") != "train" for row in rows):
        raise ValueError("split artifact contains non-target/non-TRAIN rows")
    fit_ids = [row["id"] for row in fit_rows]
    holdout_ids = [row["id"] for row in holdout_rows]
    fit_groups = {row["group"] for row in fit_rows}
    holdout_groups = {row["group"] for row in holdout_rows}
    fit_observations = {row["observation"] for row in fit_rows}
    holdout_observations = {row["observation"] for row in holdout_rows}
    if len(fit_ids) != len(set(fit_ids)) or len(holdout_ids) != len(set(holdout_ids)):
        raise ValueError("split artifact contains duplicate row IDs")
    if set(fit_ids) & set(holdout_ids):
        raise ValueError("fit and holdout row IDs overlap")
    if fit_groups & holdout_groups:
        raise ValueError("fit and holdout state groups overlap")
    if fit_observations & holdout_observations:
        raise ValueError("fit and holdout observations overlap")
    if holdout_groups & set(previously_seen_groups):
        raise ValueError("previously consumed state group entered the holdout")


def tracked_python_hashes():
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=REPO)
        .decode()
        .split("\0")
    )
    return {str(REPO / name): sha(REPO / name) for name in names if name}


def require_clean_source():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("commit the exact source before preflight/training")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def load_train_corpus(config):
    root = Path(config["corpus_root"])
    receipt = read_json(root / "result.json")
    if receipt.get("status") != "PASS_COMPLETE_NATIVE_COMMON_CORPUS":
        raise ValueError("pinned complete TRAIN corpus required")
    rows = [
        json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("TRAIN corpus contains a non-TRAIN row")
    return root, receipt, rows


def prior_sampler_ids(rows):
    # This sampler is the fixed 2048-update run recorded in the earlier native fit.
    from scripts.native_fit import sample_order

    order = sample_order(rows, UPDATES, seed=17)
    return {rows[index]["id"] for batch in order for index in batch}


def describe_split(split):
    fit_rows, holdout = split["fit_rows"], split["holdout_rows"]
    return {
        "fit_row_count": len(fit_rows),
        "fit_unique_groups": len(split["fit_groups"]),
        "fit_unique_observations": len({row["observation"] for row in fit_rows}),
        "holdout_row_count": len(holdout),
        "holdout_unique_groups": len(split["holdout_groups"]),
        "holdout_unique_observations": len({row["observation"] for row in holdout}),
        "previously_seen_group_count": len(split["previously_seen_groups"]),
        "previously_untouched_group_count": len(split["previously_untouched_groups"]),
        "fit_task_counts": split["fit_task_counts"],
        "holdout_task_counts": split["holdout_task_counts"],
        "holdout_min_rows_per_task": min(split["holdout_task_counts"].values()),
        "holdout_max_rows_per_task": max(split["holdout_task_counts"].values()),
        "fit_ordered_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in fit_rows).encode()
        ).hexdigest(),
        "holdout_ordered_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in holdout).encode()
        ).hexdigest(),
        "fit_group_set_sha256": hashlib.sha256("\n".join(split["fit_groups"]).encode()).hexdigest(),
        "holdout_group_set_sha256": hashlib.sha256(
            "\n".join(split["holdout_groups"]).encode()
        ).hexdigest(),
    }


def validate_train_text_metadata(config, split):
    root = Path(config["corpus_root"])
    observations = read_json(root / "observations.json")
    banks = read_json(root / "banks.json")
    query_map = read_json(root / "query-map.json")
    expected_state_bank = "language/train-states.npy"
    expected_query_bank = "language/queries.npy"
    if banks[expected_state_bank]["sha256"] != config["text_state_bank_sha256"]:
        raise ValueError("TRAIN state bank differs from the pinned config")
    if banks[expected_query_bank]["sha256"] != config["text_query_bank_sha256"]:
        raise ValueError("candidate query bank differs from the pinned config")
    used_queries = set()
    for row in split["fit_rows"] + split["holdout_rows"]:
        entry = observations.get(row["observation"])
        if not entry or entry.get("split") != "train" or entry.get("source") != "text":
            raise ValueError("only text observations marked TRAIN are allowed")
        features = entry.get("features", [])
        if len(features) != 1 or features[0].get("bank") != expected_state_bank:
            raise ValueError("TRAIN text state must resolve to its pinned state bank")
        used_queries.update(row["queries"])
    for key in used_queries:
        item = query_map.get(key)
        if not item or item.get("bank") != expected_query_bank:
            raise ValueError("TRAIN candidate query must resolve to the pinned query bank")
    return {
        "train_observation_metadata_count": len(
            {row["observation"] for row in split["fit_rows"] + split["holdout_rows"]}
        ),
        "train_query_ids_used": len(used_queries),
        "official_validation_rows_loaded": False,
        "official_validation_observation_payloads_loaded": False,
        "sealed_audit_loaded": False,
    }


def build_preflight(config_path, output):
    source_commit = require_clean_source()
    config = read_json(config_path)
    validate_config(config)
    source, split_id = configured_source_split(config)
    validate_output_path(config, output)
    verify_pins(config["input_pins"])
    root, corpus_receipt, rows = load_train_corpus(config)
    if (
        corpus_receipt["artifacts"].get("train.jsonl")
        != config["input_pins"][str(root / "train.jsonl")]
    ):
        raise ValueError("TRAIN corpus receipt and config hash disagree")
    prior_ids = prior_sampler_ids(rows)
    source_rows = [row for row in rows if row.get("source") == source]
    consumed = {row["id"] for row in source_rows if row["id"] in prior_ids}
    split = build_split(
        rows,
        consumed,
        seed=config["split"]["seed"],
        holdout_fraction=config["split"]["holdout_fraction_of_eligible_groups"],
        source=source,
        split_id=split_id,
    )
    description = describe_split(split)
    expected = config["split"]
    if (
        len(split["previously_untouched_groups"])
        != expected["eligible_groups_not_seen_by_prior_2048_step_sampler"]
        or len(split["holdout_groups"]) != expected["heldout_group_count"]
    ):
        raise ValueError("observed group availability differs from preregistered split")
    feature_metadata = validate_train_text_metadata(config, split)
    query_bank = Path(config["text_query_bank_path"])
    state_bank = Path(config["text_state_bank_path"])
    query_array = np.load(query_bank, mmap_mode="r", allow_pickle=False)
    state_array = np.load(state_bank, mmap_mode="r", allow_pickle=False)
    if query_array.shape != (22328, 384) or state_array.shape != (375078, 384):
        raise ValueError("pinned frozen text feature shapes changed")
    if query_array.dtype != np.float32 or state_array.dtype != np.float32:
        raise ValueError("text feature bank dtype changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    split_payload = {
        "split_id": split_id,
        "fit_ids": [row["id"] for row in split["fit_rows"]],
        "holdout_ids": [row["id"] for row in split["holdout_rows"]],
        "fit_groups": split["fit_groups"],
        "holdout_groups": split["holdout_groups"],
        "previously_seen_groups": split["previously_seen_groups"],
        "previously_untouched_groups": split["previously_untouched_groups"],
    }
    if COUNTERFACTUAL_INTERVENTION in config["evaluation"]["primary_interventions"]:
        split_payload["donor_indices"] = counterfactual_donor_indices(
            split["holdout_rows"], split_id=split_id
        )
    write_json(output / "split.json", split_payload)
    receipt = {
        "status": "PASS_TRAIN_ONLY_GROUP_PREFLIGHT",
        "source_commit": source_commit,
        "config_sha256": sha(Path(config_path)),
        "split_sha256": sha(output / "split.json"),
        "input_pins": config["input_pins"],
        "python_source_hashes": tracked_python_hashes(),
        "split": description,
        "feature_metadata": feature_metadata,
        "training_draws_planned": UPDATES * 4,
        "training_unique_rows": len(split["fit_rows"]),
        "training_repeated_draws_planned": max(0, UPDATES * 4 - len(split["fit_rows"])),
        "train_array_shapes": {
            "language_queries": list(query_array.shape),
            "language_train_states": list(state_array.shape),
        },
        "validation_jsonl_opened": False,
        "sealed_audit_opened": False,
    }
    write_json(output / "preflight.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "source_commit": source_commit,
                "config_sha256": receipt["config_sha256"],
                "split": description,
                "feature_metadata": feature_metadata,
                "training_draws_planned": receipt["training_draws_planned"],
                "training_unique_rows": receipt["training_unique_rows"],
                "training_repeated_draws_planned": receipt["training_repeated_draws_planned"],
                "validation_jsonl_opened": False,
                "sealed_audit_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


class TrainTextFeatureStore:
    """Lazy text-only view; it never reads validation observation or query rows."""

    def __init__(self, config, all_rows, fit_rows, holdout_rows, device="cuda"):
        root = Path(config["corpus_root"])
        self.observations = read_json(root / "observations.json")
        self.banks = read_json(root / "banks.json")
        query_map = read_json(root / "query-map.json")
        self.device = device
        self.state_array = np.load(
            config["text_state_bank_path"], mmap_mode="r", allow_pickle=False
        )
        self.query_array = np.load(
            config["text_query_bank_path"], mmap_mode="r", allow_pickle=False
        )
        required = {query_id for row in fit_rows + holdout_rows for query_id in row["queries"]}
        self.query_offsets = {}
        for query_id in required:
            item = query_map.get(query_id)
            if not item or item.get("bank") != "language/queries.npy":
                raise ValueError("TRAIN query feature mapping changed")
            self.query_offsets[query_id] = int(item["offset"])
        self.query_cache = {}
        self.state_cache = OrderedDict()
        self.state_cache_bytes = 0
        self.rows = {row["id"]: row for row in all_rows}

    def get(self, row):
        entry = self.observations.get(row["observation"])
        if not entry or entry.get("split") != "train" or entry.get("source") != "text":
            raise ValueError("attempted to load a non-TRAIN text state")
        if row["observation"] not in self.state_cache:
            feature = entry["features"][0]
            start, count = int(feature["offset"]), int(feature["tokens"])
            values = np.array(self.state_array[start : start + count], copy=True)
            if values.shape != (count, 384) or not np.isfinite(values).all():
                raise ValueError("invalid TRAIN text feature slice")
            x = torch.from_numpy(values).to(self.device)
            pos = positions("text", count).to(self.device)
            valid = torch.ones(count, dtype=torch.bool, device=self.device)
            blocks = [NativeTokenBlock("text", x, pos, valid)]
            size = sum(value.numel() * value.element_size() for value in (x, pos, valid))
            self.state_cache[row["observation"]] = (blocks, size)
            self.state_cache_bytes += size
            while self.state_cache_bytes > 128 * 2**20 and len(self.state_cache) > 1:
                _, (_, old_size) = self.state_cache.popitem(last=False)
                self.state_cache_bytes -= old_size
        self.state_cache.move_to_end(row["observation"])
        query_vectors = []
        for query_id in row["queries"]:
            if query_id not in self.query_cache:
                vector = np.array(self.query_array[self.query_offsets[query_id]], copy=True)
                if vector.shape != (384,) or not np.isfinite(vector).all():
                    raise ValueError("invalid TRAIN candidate-query vector")
                self.query_cache[query_id] = torch.from_numpy(vector).to(self.device)
            query_vectors.append(self.query_cache[query_id])
        return self.state_cache[row["observation"]][0], torch.stack(query_vectors)


def bootstrap_macro_deltas(rows, baseline_scores, control_scores, control_targets, seed=17):
    """Paired group bootstrap of equal-task-macro Accuracy/NLL/Brier deltas."""
    groups = sorted({row["group"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    utilities = {"baseline": [], "control": []}
    baseline_target = np.asarray([row["target_index"] for row in rows], dtype=np.int64)
    control_target = np.asarray(control_targets, dtype=np.int64)
    if control_target.shape != baseline_target.shape:
        raise ValueError("control targets must align with paired rows")
    for key, scores, target in (
        ("baseline", baseline_scores, baseline_target),
        ("control", control_scores, control_target),
    ):
        values = np.asarray(scores, dtype=np.float64)
        logp = values - np.logaddexp.reduce(values, axis=1, keepdims=True)
        probability = np.exp(logp)
        correct = values.argmax(1) == target
        nll = -logp[np.arange(len(rows)), target]
        brier = ((probability - np.eye(values.shape[1])[target]) ** 2).sum(1)
        utilities[key] = np.stack((correct, nll, brier), axis=1).astype(np.float64)
    group_index = {group: index for index, group in enumerate(groups)}
    task_group_sums = {name: {} for name in utilities}
    for task in tasks:
        indices = [i for i, row in enumerate(rows) if row["task"] == task]
        for name, values in utilities.items():
            sums = np.zeros((len(groups), 3), dtype=np.float64)
            counts = np.zeros(len(groups), dtype=np.float64)
            for index in indices:
                group = group_index[rows[index]["group"]]
                sums[group] += values[index]
                counts[group] += 1
            task_group_sums[name][task] = (sums, counts)
    rng = np.random.default_rng(seed)
    samples = {
        name: np.empty((BOOTSTRAP_REPLICATES, 3), dtype=np.float64)
        for name in ("baseline", "control")
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(groups), size=len(groups))
        multiplicities = np.bincount(selected, minlength=len(groups))
        for name in samples:
            per_task = []
            for task in tasks:
                sums, counts = task_group_sums[name][task]
                denominator = float(counts @ multiplicities)
                if denominator <= 0:
                    raise ValueError("bootstrap task has no sampled rows")
                per_task.append((sums.T @ multiplicities) / denominator)
            samples[name][replicate] = np.mean(per_task, axis=0)
    differences = samples["control"] - samples["baseline"]
    point = np.asarray(
        [
            np.mean(
                [
                    utilities[name][
                        [i for i, row in enumerate(rows) if row["task"] == task], j
                    ].mean()
                    for task in tasks
                ]
            )
            for name in ("baseline", "control")
            for j in range(3)
        ]
    ).reshape(2, 3)
    delta_point = point[1] - point[0]
    return {
        "cluster": "underlying TRAIN text state group",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "interval": "two-sided 95% percentile paired group bootstrap",
        "metrics": {
            metric: {
                "difference_control_minus_baseline": float(delta_point[index]),
                "interval95": [
                    float(value) for value in np.percentile(differences[:, index], [2.5, 97.5])
                ],
            }
            for index, metric in enumerate(("accuracy", "nll", "brier"))
        },
    }


def train(config_path, output, preflight_path, preflight_sha256):
    source_commit = require_clean_source()
    config = read_json(config_path)
    validate_config(config)
    source, split_id = configured_source_split(config)
    validate_output_path(config, output)
    if sha(Path(preflight_path) / "preflight.json") != preflight_sha256:
        raise ValueError("preflight receipt hash mismatch")
    preflight = read_json(Path(preflight_path) / "preflight.json")
    if (
        preflight["status"] != "PASS_TRAIN_ONLY_GROUP_PREFLIGHT"
        or preflight["source_commit"] != source_commit
        or preflight["config_sha256"] != sha(Path(config_path))
    ):
        raise ValueError("preflight/source/config does not match this run")
    if sha(Path(preflight_path) / "split.json") != preflight.get("split_sha256"):
        raise ValueError("preflight split artifact hash mismatch")
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    memory, utilization = [int(value.strip()) for value in usage.split(",")]
    if memory > 512 or utilization > 5:
        raise RuntimeError("GPU occupied; wait for the cooperative local GPU lease")
    verify_pins(config["input_pins"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = {
        "status": "RUNNING",
        "run_id": config["id"],
        "source_commit": source_commit,
        "config_sha256": sha(Path(config_path)),
        "preflight_sha256": preflight_sha256,
        "preflight_split_sha256": sha(Path(preflight_path) / "split.json"),
        "command": sys.argv,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "official_validation_loaded": False,
        "sealed_audit_loaded": False,
        "cloud_cost": 0,
        "gpu_preflight": usage,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "result.json", result)
    try:
        root, corpus_receipt, rows = load_train_corpus(config)
        split_payload = read_json(Path(preflight_path) / "split.json")
        if split_payload.get("split_id") != split_id:
            raise ValueError("preflight split identity does not match the configured source")
        by_id = {row["id"]: row for row in rows}
        fit_rows = [by_id[ident] for ident in split_payload["fit_ids"]]
        holdout = [by_id[ident] for ident in split_payload["holdout_ids"]]
        validate_split_membership(
            fit_rows,
            holdout,
            split_payload.get("previously_seen_groups", []),
            source,
        )
        if (
            len(fit_rows) != preflight["split"]["fit_row_count"]
            or len(holdout) != preflight["split"]["holdout_row_count"]
        ):
            raise ValueError("preflight row counts changed")
        order = balanced_order(fit_rows, UPDATES, seed=17, split_id=split_id)
        draws = [index for batch in order for index in batch]
        sampling = {
            "steps": len(order),
            "microbatches_per_update": 4,
            "draws": len(draws),
            "unique_rows_drawn": len({fit_rows[index]["id"] for index in draws}),
            "repeated_draws": len(draws) - len({fit_rows[index]["id"] for index in draws}),
            "order_sha256": hashlib.sha256(
                "\n".join(
                    ",".join(fit_rows[index]["id"] for index in batch) for batch in order
                ).encode()
            ).hexdigest(),
            "batch_row_ids": [[fit_rows[index]["id"] for index in batch] for batch in order],
        }
        write_json(output / "sampling.json", sampling)
        store = TrainTextFeatureStore(config, rows, fit_rows, holdout)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
        model = NativeCandidateReadout().to("cuda")
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = 195_365_529 + sum(p.numel() for p in model.parameters())
        if total_parameters > 200_000_000:
            raise ValueError("prequantization parameter ceiling exceeded")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        checkpoint_hashes = {}
        result.update(
            status="TRAINING",
            run_id=config["id"],
            source=source,
            variant=(
                "text-only native readout; frozen pretrained text states and candidate features"
            ),
            initialization="fresh NativeCandidateReadout random seed 17",
            updates_planned=UPDATES,
            trainable_parameters=trainable_parameters,
            total_prequantization_parameters=total_parameters,
            train_rows=len(fit_rows),
            holdout_rows=len(holdout),
            holdout_groups=len(split_payload["holdout_groups"]),
            unique_queries_loaded=len(store.query_offsets),
            state_cache_max_bytes=128 * 2**20,
            device=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            schedule="AdamW lr 0.001; 62-update warmup; cosine to zero at update 2048",
            checkpoint_policy="fixed terminal only; no holdout checkpoint selection",
            training_draws={
                key: value for key, value in sampling.items() if key != "batch_row_ids"
            },
            feature_cache_bytes_hashed=True,
            data_artifact_hashes=config["input_pins"],
        )
        write_json(output / "result.json", result)
        history_path = output / "training-history.jsonl"
        losses = []
        corrects = []
        update_seconds = []
        with history_path.open("x", encoding="utf-8") as log:
            for step, batch in enumerate(order, start=1):
                tick = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                batch_losses = []
                batch_correct = 0
                for index in batch:
                    row = fit_rows[index]
                    blocks, queries = store.get(row)
                    permutation = option_order(row, step)
                    target = permutation.index(row["target_index"])
                    scores = model(blocks, queries[permutation])
                    loss = -scores.log_softmax(0)[target]
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite loss at step {step}")
                    (loss / 4).backward()
                    batch_losses.append(float(loss.detach()))
                    batch_correct += int(scores.detach().argmax().item() == target)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise ValueError(f"nonfinite gradient at step {step}")
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                update_seconds.append(elapsed)
                losses.append(float(np.mean(batch_losses)))
                corrects.append(batch_correct / 4)
                item = {
                    "step": step,
                    "rolling_128_update_train_ce": float(np.mean(losses[-128:])),
                    "rolling_128_update_train_accuracy": float(np.mean(corrects[-128:])),
                    "lr": lr,
                    "gradient_norm_before_clip": float(gradient_norm),
                    "seconds": elapsed,
                    "train_only": True,
                }
                log.write(json.dumps(item, allow_nan=False) + "\n")
                if step % 32 == 0:
                    log.flush()
                    result.update(
                        status="TRAINING",
                        step=step,
                        training_seconds=float(sum(update_seconds)),
                        median_update_seconds=float(np.median(update_seconds[-128:])),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                    write_json(output / "result.json", result)
                    print(
                        "step",
                        step,
                        "train_ce",
                        item["rolling_128_update_train_ce"],
                        "median_update_s",
                        result["median_update_seconds"],
                        flush=True,
                    )
                if step % CHECKPOINT_INTERVAL == 0 or step == UPDATES:
                    checkpoint = output / f"checkpoint-{step:04}.pt"
                    torch.save({"model": model.state_dict(), "step": step, "seed": 17}, checkpoint)
                    checkpoint_hashes[str(step)] = sha(checkpoint)
                    result["checkpoint_sha256_by_step"] = checkpoint_hashes.copy()
                    result["last_checkpoint_step"] = step
                    write_json(output / "result.json", result)
                if time.perf_counter() - start > 7200:
                    raise RuntimeError("two-hour local runtime guard exceeded")
                if torch.cuda.max_memory_allocated() > 6 * 2**30:
                    raise RuntimeError("6 GiB CUDA allocation guard exceeded")
                if shutil.disk_usage(output).free < 2 * 2**30:
                    raise RuntimeError("2 GiB disk reserve guard exceeded")
        if step != UPDATES:
            raise RuntimeError("fixed 2048 updates not completed")

        use_counterfactual = (
            COUNTERFACTUAL_INTERVENTION in config["evaluation"]["primary_interventions"]
        )
        donor_indices = (
            {int(key): int(value) for key, value in split_payload["donor_indices"].items()}
            if use_counterfactual
            else {}
        )
        score_width = max(len(row["options"]) for row in holdout)
        normal_scores = np.full((len(holdout), score_width), -1e30, dtype=np.float32)
        zero_scores = np.full_like(normal_scores, -1e30)
        swapped_scores = np.full_like(normal_scores, -1e30) if use_counterfactual else None
        swap_rows = []
        prediction_lines = []
        model.eval()
        with torch.no_grad():
            for index, row in enumerate(holdout):
                blocks, queries = store.get(row)
                normal = model(blocks, queries).cpu().numpy()
                zero_blocks = [
                    NativeTokenBlock(
                        block.kind, torch.zeros_like(block.values), block.positions, block.valid
                    )
                    for block in blocks
                ]
                zero = model(zero_blocks, queries).cpu().numpy()
                normal_scores[index, : len(normal)] = normal
                zero_scores[index, : len(zero)] = zero
                prediction = {
                    "id": row["id"],
                    "state_group": row["group"],
                    "task": row["task"],
                    "target_id": row["option_ids"][row["target_index"]],
                    "prediction_id": row["option_ids"][int(normal.argmax())],
                    "normal_scores": normal.tolist(),
                    "zero_state_scores": zero.tolist(),
                }
                if use_counterfactual:
                    donor = holdout[donor_indices[index]]
                    donor_blocks, _ = store.get(donor)
                    swapped = model(donor_blocks, queries).cpu().numpy()
                    donor_answer = donor["option_ids"][donor["target_index"]]
                    if donor_answer not in row["option_ids"]:
                        raise ValueError("counterfactual donor answer is absent from row options")
                    swapped_target = row["option_ids"].index(donor_answer)
                    swapped_scores[index, : len(swapped)] = swapped
                    changed = copy(row)
                    changed["target_index"] = swapped_target
                    changed["group"] = donor["group"]
                    changed["observation"] = donor["observation"]
                    swap_rows.append(changed)
                    prediction.update(
                        donor_id=donor["id"],
                        donor_group=donor["group"],
                        counterfactual_state_scores=swapped.tolist(),
                        counterfactual_target_id=donor_answer,
                    )
                prediction_lines.append(json.dumps(prediction, ensure_ascii=False, allow_nan=False))
                if (index + 1) % 128 == 0:
                    print("holdout inference", index + 1, "/", len(holdout), flush=True)
        torch.cuda.synchronize()
        prediction_arrays = {"normal": normal_scores, "zero_state": zero_scores}
        if use_counterfactual:
            prediction_arrays["counterfactual_state"] = swapped_scores
        np.savez_compressed(output / "holdout-predictions.npz", **prediction_arrays)
        (output / "predictions.jsonl").write_text(
            "\n".join(prediction_lines) + "\n", encoding="utf-8"
        )
        normal_metrics = summarize(holdout, normal_scores)
        zero_metrics = summarize(holdout, zero_scores)
        input_use = {
            "normal": normal_metrics,
            "zero_state": zero_metrics,
            "primary_interventions": config["evaluation"]["primary_interventions"],
            "normal_vs_zero_state_macro_group_bootstrap": bootstrap_macro_deltas(
                holdout, normal_scores, zero_scores, [row["target_index"] for row in holdout]
            ),
        }
        if use_counterfactual:
            input_use.update(
                counterfactual_state_swap=summarize(swap_rows, swapped_scores),
                counterfactual_targets_mapped_by_option_id=True,
                normal_vs_counterfactual_swap_group_bootstrap=bootstrap_macro_deltas(
                    holdout,
                    normal_scores,
                    swapped_scores,
                    [row["target_index"] for row in swap_rows],
                ),
            )
        write_json(output / "holdout-metrics.json", input_use)
        final_checkpoint = output / f"checkpoint-{UPDATES:04}.pt"
        model.load_state_dict(
            torch.load(final_checkpoint, map_location="cuda", weights_only=True)["model"]
        )
        reload_scores = np.full_like(normal_scores, -1e30)
        with torch.no_grad():
            for index, row in enumerate(holdout):
                blocks, queries = store.get(row)
                scores = model(blocks, queries).cpu().numpy()
                reload_scores[index, : len(scores)] = scores
        torch.cuda.synchronize()
        np.testing.assert_array_equal(normal_scores, reload_scores)
        result.update(
            status="PASS_TRAIN_ONLY_GROUP_HOLDOUT_COMPLETE",
            source=source,
            split_id=split_id,
            step=UPDATES,
            best_step=None,
            selected_step=UPDATES,
            selection_rule="fixed terminal update; no holdout checkpoint selection",
            official_validation_loaded=False,
            sealed_audit_loaded=False,
            training_seconds=float(sum(update_seconds)),
            evaluation_seconds=float(time.perf_counter() - start - sum(update_seconds)),
            median_update_seconds=float(np.median(update_seconds)),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            train_rolling_128_ce_first=float(np.mean(losses[:128])),
            train_rolling_128_ce_last=float(np.mean(losses[-128:])),
            train_rolling_128_accuracy_first=float(np.mean(corrects[:128])),
            train_rolling_128_accuracy_last=float(np.mean(corrects[-128:])),
            holdout_metrics_sha256=sha(output / "holdout-metrics.json"),
            prediction_artifact_sha256=sha(output / "holdout-predictions.npz"),
            checkpoint_reloaded_exactly=True,
            reload_max_abs=0.0,
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - start,
        )
        write_json(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        result.update(
            status="BLOCKED_OR_FAILED",
            error=repr(error),
            traceback=traceback.format_exc(),
            step=result.get("step", 0),
            finished_utc=datetime.now(UTC).isoformat(),
        )
        write_json(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args()
    if args.phase == "preflight":
        if args.preflight or args.preflight_sha256:
            raise ValueError("preflight phase does not consume another receipt")
        build_preflight(args.config, args.output)
    else:
        if not args.preflight or not args.preflight_sha256:
            raise ValueError("train requires preflight path and SHA-256")
        train(args.config, args.output, args.preflight, args.preflight_sha256)


if __name__ == "__main__":
    main()
