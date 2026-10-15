"""File binding for the fixed post-fit comparison; no model or optimizer code.

Only synthetic files exercise successful transfer reads until both real fits
complete. The later inference runner must also audit the timeline/selector and
validate complete row groups before scoring. This module does not do that work.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts import condition_relevance_postfit as contract

CONFIG = "configs/pretrained_reuse/condition_relevance_fit_v1.json"
PROTOCOL = "docs/pretrained_reuse/R2_CONDITION_RELEVANCE_FIT_PROTOCOL_20261015.md"
DEVELOPMENT = (
    "binding_train",
    "numeric_train",
    "legacy_train",
    "validation",
    "old_validation",
    "legacy_validation",
)
POSTFIT_MODULES = {
    "condition_relevance_postfit",
    "condition_relevance_postfit_io",
    "condition_relevance_postfit_evidence",
    "condition_relevance_inference",
    "condition_relevance_transfer",
    "prepare_condition_relevance_comparison",
    "evaluate_condition_relevance_transfer",
}
ARTIFACTS = (
    set(contract.COMMON_ARTIFACTS)
    | {
        "best-delta.pt",
        "final-delta.pt",
        "effective-config.json",
        "history.json",
        "updates.jsonl",
        "uncertainty.json",
    }
    | {f"step-{step}-delta.pt" for step in contract.STEPS}
    | {
        f"{prefix}-{suffix}"
        for prefix in [
            *(
                f"{part}-{step}"
                for part in ("validation", "fixed-train")
                for step in contract.STEPS
            ),
            "reload-validation",
            "reload-fixed-train",
        ]
        for suffix in ("metrics.json", "predictions.jsonl", "scores.npy")
    }
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _simple_name(value):
    return (
        isinstance(value, str)
        and value not in ("", ".", "..")
        and not any(c in value for c in "/\\:\0")
    )


def _verified_bytes(path, expected):
    _require(contract._hash(expected), "valid expected SHA256 required")
    raw = _no_redirect(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected, "file hash mismatch: " + str(path))
    return raw


def _json(path, expected):
    return json.loads(_verified_bytes(path, expected))


def _verify_stream(path, expected):
    _require(contract._hash(expected), "valid expected SHA256 required")
    digest = hashlib.sha256()
    with _no_redirect(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    _require(digest.hexdigest() == expected, "file hash mismatch: " + str(path))


def _no_redirect(path):
    declared = Path(path).absolute()
    _require(declared.resolve() == declared, "linked or redirected path forbidden: " + str(path))
    return declared


def _normalized_pins(pins):
    result = {_no_redirect(p): h for p, h in pins.items()}
    _require(len(result) == len(pins), "aliased paths in pin map")
    return result


def verify_execution_source(source_root, entry_file, entry_name, pins, *, modules=None):
    """Bind actual local module origins, not just unused files, to pinned bytes."""
    source_root = _no_redirect(source_root)
    entry = _no_redirect(entry_file)
    _require(
        entry == source_root / "scripts" / (entry_name + ".py"),
        "executing entrypoint differs from registered source",
    )
    pins = _normalized_pins(pins)
    for path, expected in pins.items():
        _require(
            path.is_relative_to(source_root) and path.suffix == ".py",
            "evaluation source pin outside declared Python scope",
        )
        _verify_stream(path, expected)
    origins = {"__main__": entry}
    for name, module in list((sys.modules if modules is None else modules).items()):
        if name in POSTFIT_MODULES:
            expected = source_root / "scripts" / name
        elif name.split(".")[0] in ("scripts", "tiny_omni_decision"):
            parent = source_root / "src" if name.startswith("tiny_omni_decision") else source_root
            expected = parent.joinpath(*name.split("."))
        else:
            continue
        actual = getattr(module, "__file__", None)
        if actual is None:
            locations = [_no_redirect(p) for p in getattr(module, "__path__", [])]
            _require(locations == [expected], "imported namespace path differs: " + name)
            continue
        actual = _no_redirect(actual)
        _require(
            actual in (expected.with_suffix(".py"), expected / "__init__.py"),
            "imported module origin differs from registered source: " + name,
        )
        origins[name] = actual
    for name, path in origins.items():
        _require(path in pins, "executing module source pin missing: " + name)
    return {name: {"path": str(path), "sha256": pins[path]} for name, path in origins.items()}


def capture_execution_source(source_root, entry_file, entry_name):
    """Capture committed evaluation code and verify the actual current imports."""
    source_root = _no_redirect(source_root)
    _require(
        _no_redirect(entry_file) == source_root / "scripts" / (entry_name + ".py"),
        "executing entrypoint differs from registered source",
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "*.py"],
        cwd=source_root, check=True, capture_output=True,
    )
    files = subprocess.check_output(
        ["git", "ls-files", "--", "*.py"], cwd=source_root, text=True
    ).splitlines()
    pins = {
        str(_no_redirect(source_root / p)): hashlib.sha256(
            _no_redirect(source_root / p).read_bytes()
        ).hexdigest()
        for p in files
    }
    origins = verify_execution_source(source_root, entry_file, entry_name, pins)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    return {"code_pins": pins, "module_origins": origins, "evaluation_source_commit": commit}


def verify_pair_files(run_directories, result_hashes, source_root):
    """Bind terminal metadata to bytes without opening transfer/calibration/audit."""
    _require(set(run_directories) == {"ce", "relevance"}, "two declared run directories required")
    _require(set(result_hashes) == set(run_directories), "two expected result hashes required")
    directories = {a: _no_redirect(p) for a, p in run_directories.items()}
    _require(len(set(directories.values())) == 2, "distinct run directories required")
    runs = {a: _json(p / "result.json", result_hashes[a]) for a, p in directories.items()}
    # Do not touch effective config, corpus, model or other outputs before this.
    contract.require_completed_pair(runs["ce"], runs["relevance"])
    for a, run in runs.items():
        _require(set(run["artifacts"]) <= ARTIFACTS, "unregistered run artifact")
        _require(_no_redirect(run["command"][-1]) == directories[a], "run output path mismatch")
    effective = {
        a: _json(
            directories[a] / "effective-config.json", run["artifacts"]["effective-config.json"]
        )
        for a, run in runs.items()
    }
    frozen = contract.freeze_comparison(
        runs["ce"], runs["relevance"], result_hashes, effective_configs=effective
    )
    source_root = _no_redirect(source_root)
    config = _json(source_root / CONFIG, contract.CONFIG_SHA)
    _require(config == runs["ce"]["config"], "actual fixed config differs from effective config")
    code_pins = _normalized_pins(runs["ce"]["code_pins"])
    _require(
        code_pins.get(source_root / CONFIG) == contract.CONFIG_SHA, "config source pin missing"
    )
    runner = source_root / "scripts/train_condition_relevance.py"
    _require(runner in code_pins, "executed runner source pin missing")
    _require(
        all(_no_redirect(run["command"][0]) == runner for run in runs.values()),
        "executed runner differs from registered source",
    )
    for path in code_pins:
        _require(
            path.is_relative_to(source_root)
            and (
                path.suffix == ".py"
                or path == source_root / "pyproject.toml"
                or path == source_root / CONFIG
                or path == source_root / PROTOCOL
            ),
            "source pin outside declared code scope",
        )
    corpus_root, model_root = _no_redirect(config["corpus"]), _no_redirect(config["model"])
    corpus = _json(corpus_root / "result.json", contract.CORPUS_SHA)
    base = _json(model_root.parent / "download.json", config["model_manifest_sha256"])
    _require(base.get("revision") == contract.REVISION, "base revision changed")
    _require(contract._hash_map(base.get("hashes")), "base/processor hashes missing")
    _require(
        all(_simple_name(n) for n in base["hashes"]), "base artifact path not a simple filename"
    )
    expected_inputs = {
        corpus_root / "result.json": contract.CORPUS_SHA,
        model_root.parent / "download.json": config["model_manifest_sha256"],
        **{model_root / n: h for n, h in base["hashes"].items()},
        **{corpus_root / (n + ".jsonl"): corpus["artifacts"][n + ".jsonl"] for n in DEVELOPMENT},
        corpus_root / "selection-schema.json": corpus["artifacts"]["selection-schema.json"],
    }
    _require(
        _normalized_pins(runs["ce"]["input_pins"]) == expected_inputs,
        "undeclared or missing fit input pins",
    )
    payload = contract.transfer_contract(
        frozen,
        runs["ce"],
        runs["relevance"],
        result_hashes,
        corpus,
        contract.CORPUS_SHA,
        effective_configs=effective,
    )
    for pins in (code_pins, expected_inputs):
        for path, expected in pins.items():
            _verify_stream(path, expected)
    for arm, run in runs.items():
        for name, expected in run["artifacts"].items():
            _verify_stream(directories[arm] / name, expected)
    return {
        "frozen": frozen,
        "runs": runs,
        "effective_configs": effective,
        "run_directories": directories,
        "source_root": source_root,
        "corpus_root": corpus_root,
        "model_root": model_root,
        "payload": payload,
    }


def _outside_protected(checked, path):
    resolved = _no_redirect(path)
    protected = [
        *checked["run_directories"].values(),
        checked["source_root"],
        checked["corpus_root"],
        checked["model_root"].parent,
    ]
    _require(
        not any(resolved.is_relative_to(p) for p in protected),
        "freeze must be outside protected paths",
    )
    return resolved


def write_freeze(checked, destination):
    """Persist a new identity record; never overwrite or write inside old artifacts."""
    destination = _outside_protected(checked, destination)
    raw = (json.dumps(checked["frozen"], sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_frozen_transfer(freeze_path, freeze_sha256, run_directories, result_hashes, source_root):
    """Read only the frozen transfer bytes after rechecking both completed arms.

    No evaluation, calibration or model selection is performed here. Callers
    must validate all rows and satisfy the separate fit-evidence audit first.
    """
    checked = verify_pair_files(run_directories, result_hashes, source_root)
    freeze_path = _outside_protected(checked, freeze_path)
    frozen = _json(freeze_path, freeze_sha256)
    _require(frozen == checked["frozen"], "saved frozen identities changed")
    payload = checked["payload"]
    return _verified_bytes(checked["corpus_root"] / payload["filename"], payload["sha256"])
