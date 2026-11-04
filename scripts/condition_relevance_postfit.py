"""Pure post-fit metadata contract; callers must verify actual file hashes.

No file/model/data loading occurs here. A successful contract is not itself a
byte-integrity check, a transfer evaluation, or a quality/adoption decision.
"""

import hashlib
import json
import re

SOURCE = "6fb617f233752f582af7daf0dd131ba7cc281a5c"
CONFIG_SHA = "942e86187fa587689165269b92e424d70fe5d615fbc3301342af48204fe1b5db"
CORPUS_SHA = "7593e36769996d07e3654694013a989a8cdcd8a4b58442a29bde4209feb26964"
REVISION = "262ae02f29173eec1c250f90804dc7edc677dcff"
STEPS = (0, 256, 512, 768, 1024)
COMMON_ARTIFACTS = (
    "initial-delta.pt",
    "initial-retention-scores.npy",
    "sample-order.json",
    "sample-accounting.json",
    "token-preflight.json",
    "step-0-delta.pt",
    "validation-0-metrics.json",
    "validation-0-predictions.jsonl",
    "validation-0-scores.npy",
    "fixed-train-0-metrics.json",
    "fixed-train-0-predictions.jsonl",
    "fixed-train-0-scores.npy",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _hash_map(value):
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(k, str) and bool(k) and _hash(v) for k, v in value.items())
    )


def _count(value, expected):
    return type(value) is int and value == expected


def _completed(run, arm, coefficient):
    _require(isinstance(run, dict), "run metadata required")
    _require(run.get("status") == "PASS_MATCHED_RELEVANCE_FIT", "completed main fit required")
    _require(run.get("smoke") is False and _count(run.get("steps"), 1024), "full1024 fit required")
    _require(
        run.get("arm") == arm and run.get("relevance_coefficient") == coefficient,
        "registered arm/coefficient required",
    )
    for key in ("frozen_and_sources_unchanged", "native_forward_equivalence"):
        _require(run.get(key) is True, key + " must pass")
    for key in ("sealed_audit_loaded", "transfer_loaded", "exact_resume"):
        _require(run.get(key) is False, key + " must remain false")
    _require(
        run.get("selected_reload_exact") == ["validation", "fixed-train"],
        "both exact selected reloads required",
    )
    _require(
        run.get("source_commit") == SOURCE and run.get("config_sha256") == CONFIG_SHA,
        "original matched experiment source/config required",
    )
    config = run.get("config")
    _require(isinstance(config, dict), "config required")
    for key, expected in dict(
        seed=17,
        steps=1024,
        evaluation_interval=256,
        questions_per_update=[8, 8, 2],
        relevance_coefficients={"ce": 0.0, "relevance": 0.1},
        corpus_sha256=CORPUS_SHA,
        revision=REVISION,
    ).items():
        _require(config.get(key) == expected, "fixed config changed: " + key)
    for key in ("input_pins", "code_pins", "artifacts"):
        _require(_hash_map(run.get(key)), "complete hash map required: " + key)
    _require(_hash(run.get("frozen_sha256")), "frozen parameter hash required")
    best = run.get("best_step")
    _require(type(best) is int and best in STEPS, "scheduled selected step required")
    artifacts = run["artifacts"]
    required = set(COMMON_ARTIFACTS) | {
        "effective-config.json",
        "best-delta.pt",
        "final-delta.pt",
        "step-1024-delta.pt",
        f"step-{best}-delta.pt",
    }
    _require(required <= set(artifacts), "required initial/selected/endpoint artifact missing")
    _require(
        _hash(run.get("selected_sha256"))
        and run["selected_sha256"]
        == artifacts["best-delta.pt"]
        == artifacts[f"step-{best}-delta.pt"],
        "selected weight identity mismatch",
    )
    consumed, selected = run.get("consumed"), run.get("selected_consumed")
    _require(isinstance(consumed, dict) and isinstance(selected, dict), "accounting required")
    for key, expected in dict(
        consumed=18432,
        unique_questions=18432,
        repeated_questions=0,
        unique_states=2626,
        native_assets=0,
    ).items():
        _require(_count(consumed.get(key), expected), "full-run accounting changed: " + key)
    for key, expected in dict(
        consumed=best * 18, unique_questions=best * 18, repeated_questions=0
    ).items():
        _require(_count(selected.get(key), expected), "selected accounting changed: " + key)


def require_completed_pair(ce, relevance):
    """Reject nonterminal or unmatched metadata, independently of quality gains."""
    _completed(ce, "ce", 0.0)
    _completed(relevance, "relevance", 0.1)
    for key in (
        "source_commit",
        "config_sha256",
        "config",
        "input_pins",
        "code_pins",
        "frozen_sha256",
    ):
        _require(ce[key] == relevance[key], "pair mismatch: " + key)
    for name in COMMON_ARTIFACTS:
        _require(
            ce["artifacts"][name] == relevance["artifacts"][name],
            "initial conditions mismatch: " + name,
        )


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def require_matching_effective(ce, relevance, effective_configs):
    """Compare effective contents; their raw-byte hash binding is the caller's job."""
    _require(
        isinstance(effective_configs, dict) and set(effective_configs) == {"ce", "relevance"},
        "both effective configurations required",
    )
    fields = {
        "config",
        "config_sha256",
        "environment",
        "input_pins",
        "code_pins",
        "trainable_names",
        "frozen_sha256",
        "smoke",
        "arm",
        "relevance_coefficient",
        "auxiliary_parameters",
        "initial_state",
        "command",
    }
    common = {}
    for arm, run in [("ce", ce), ("relevance", relevance)]:
        value = effective_configs[arm]
        _require(isinstance(value, dict) and set(value) == fields, "effective schema changed")
        for key in (
            "config",
            "config_sha256",
            "input_pins",
            "code_pins",
            "frozen_sha256",
            "smoke",
            "arm",
            "relevance_coefficient",
            "command",
        ):
            _require(value[key] == run.get(key), "effective/terminal mismatch: " + key)
        _require(value["smoke"] is False, "effective smoke forbidden")
        _require(value["initial_state"] == "pinned_pretrained_no_delta", "fresh base required")
        _require(
            _count(value["auxiliary_parameters"], 385)
            and value["auxiliary_parameters"] == value["config"].get("auxiliary_parameters"),
            "fixed auxiliary scope required",
        )
        environment = value["environment"]
        _require(
            isinstance(environment, dict)
            and set(environment)
            == {"python", "torch", "cuda", "gpu", "dtype", "attention", "packages"}
            and environment.get("dtype") == "torch.float32"
            and environment.get("attention") == "eager"
            and all(
                isinstance(environment[k], str) and environment[k]
                for k in environment
                if k != "packages"
            )
            and isinstance(environment["packages"], dict)
            and bool(environment["packages"]),
            "fixed effective environment required",
        )
        names = value["trainable_names"]
        _require(
            isinstance(names, list)
            and bool(names)
            and all(
                isinstance(n, str) and n.startswith(("deberta.encoder.", "pooler.", "classifier."))
                for n in names
            )
            and len(names) == len(set(names)),
            "unique encoder/pooler/classifier trainable scope required",
        )
        command = value["command"]
        _require(
            isinstance(command, list)
            and len(command) == 5
            and all(isinstance(v, str) and bool(v) for v in command)
            and command[0].replace("\\", "/").endswith("/scripts/train_condition_relevance.py")
            and command[1:4] == ["--arm", arm, "--output"],
            "fixed main-fit command required; overrides forbidden",
        )
        common[arm] = {
            k: v for k, v in value.items() if k not in ("arm", "relevance_coefficient", "command")
        }
    _require(common["ce"] == common["relevance"], "common effective configuration mismatch")
    ce_command = effective_configs["ce"]["command"]
    aux_command = effective_configs["relevance"]["command"]
    _require(
        ce_command[0] == aux_command[0] and ce_command[-1] != aux_command[-1],
        "same runner and separate output directories required",
    )
    return common["ce"]


def freeze_comparison(ce, relevance, result_hashes, *, effective_configs):
    """Build all prescribed roles after caller verifies terminal result bytes."""
    require_completed_pair(ce, relevance)
    common_effective = require_matching_effective(ce, relevance, effective_configs)
    _require(
        _hash_map(result_hashes)
        and set(result_hashes) == {"ce", "relevance"}
        and result_hashes["ce"] != result_hashes["relevance"],
        "two distinct terminal result hashes required",
    )
    roles = {}

    def add(name, arm, metadata, filename, step):
        roles[name] = dict(
            arm=arm, filename=filename, step=step, sha256=metadata["artifacts"][filename]
        )

    add("initial", "ce", ce, "step-0-delta.pt", 0)
    for arm, metadata in [("ce", ce), ("relevance", relevance)]:
        add(arm + "_endpoint", arm, metadata, "step-1024-delta.pt", 1024)
        add(arm + "_selected", arm, metadata, "best-delta.pt", metadata["best_step"])
    return dict(
        schema="condition-relevance-frozen-comparison-v1",
        source_commit=SOURCE,
        config_sha256=CONFIG_SHA,
        corpus_sha256=CORPUS_SHA,
        base_revision=REVISION,
        result_hashes=dict(result_hashes),
        effective_config_hashes={
            arm: metadata["artifacts"]["effective-config.json"]
            for arm, metadata in [("ce", ce), ("relevance", relevance)]
        },
        common_effective_sha256=_digest(common_effective),
        roles=roles,
        input_pin_manifest_sha256=_digest(ce["input_pins"]),
        code_pin_manifest_sha256=_digest(ce["code_pins"]),
        common_artifact_hashes={name: ce["artifacts"][name] for name in COMMON_ARTIFACTS},
    )


def transfer_contract(
    frozen, ce, relevance, result_hashes, corpus, corpus_sha256, *, effective_configs
):
    """Return one allowed payload identity; perform no file/data access.

    The inference caller must check the frozen JSON's hash and actual run,
    weight, source, base/processor and payload bytes before consuming records.
    Completion and immutable role checks precede corpus metadata inspection.
    """
    expected = freeze_comparison(ce, relevance, result_hashes, effective_configs=effective_configs)
    _require(frozen == expected, "frozen comparison identity changed")
    _require(corpus_sha256 == CORPUS_SHA, "frozen corpus hash mismatch")
    _require(isinstance(corpus, dict), "corpus manifest required")
    _require(
        corpus.get("status") == "PASS_FROZEN_CONDITION_RELEVANCE"
        and corpus.get("sealed_audit_loaded") is False
        and corpus.get("new_transfer_model_evaluated") is False,
        "original unevaluated transfer preparation manifest required",
    )
    for flag in (
        "old_transfer_payload_opened",
        "old_state_partitions_opened",
        "calibration_loaded_for_fit",
    ):
        _require(corpus.get(flag) is False, flag + " must remain explicitly false")
    artifacts = corpus.get("artifacts")
    _require(
        isinstance(artifacts, dict) and _hash(artifacts.get("transfer.jsonl")),
        "registered transfer payload hash missing",
    )
    return dict(
        filename="transfer.jsonl", sha256=artifacts["transfer.jsonl"], rows=16384, states=256
    )
