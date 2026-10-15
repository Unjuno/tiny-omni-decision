import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_condition_relevance_postfit import effective_pair, pair

from scripts import condition_relevance_postfit as contract


@pytest.fixture
def execution_source(tmp_path):
    root = tmp_path / "repo"
    paths = {
        "entry": root / "scripts/evaluate_condition_relevance_transfer.py",
        "scripts.helper": root / "scripts/helper.py",
        "tiny_omni_decision": root / "src/tiny_omni_decision/__init__.py",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned fixture\n", encoding="utf-8")
    modules = {
        "scripts": SimpleNamespace(__file__=None, __path__=[str(root / "scripts")]),
        **{
            name: SimpleNamespace(__file__=str(path))
            for name, path in paths.items()
            if name != "entry"
        },
    }
    return root, paths, {str(p): sha(p) for p in paths.values()}, modules


def bind_execution(fixture):
    root, paths, pins, modules = fixture
    return io().verify_execution_source(
        root, paths["entry"], "evaluate_condition_relevance_transfer", pins, modules=modules
    )


def test_executing_source_and_imported_modules_match_pins(execution_source):
    evidence = bind_execution(execution_source)
    assert set(evidence) == {"__main__", "scripts.helper", "tiny_omni_decision"}


def test_copied_entrypoint_rejected_even_when_original_is_unchanged(execution_source, tmp_path):
    _, paths, _, _ = execution_source
    copy = tmp_path / paths["entry"].name
    copy.write_bytes(paths["entry"].read_bytes())
    paths["entry"] = copy
    with pytest.raises(ValueError, match="executing entrypoint"):
        bind_execution(execution_source)


@pytest.mark.parametrize("name", ["scripts.helper", "tiny_omni_decision"])
def test_stale_import_origin_rejected(execution_source, tmp_path, name):
    _, paths, _, modules = execution_source
    stale = tmp_path / paths[name].name
    stale.write_bytes(paths[name].read_bytes())
    modules[name].__file__ = str(stale)
    with pytest.raises(ValueError, match="imported module origin"):
        bind_execution(execution_source)


def test_unpinned_local_import_rejected(execution_source):
    _, paths, pins, _ = execution_source
    del pins[str(paths["scripts.helper"])]
    with pytest.raises(ValueError, match="module source pin"):
        bind_execution(execution_source)


def test_changed_executing_module_bytes_rejected(execution_source):
    _, paths, _, _ = execution_source
    paths["scripts.helper"].write_text("# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        bind_execution(execution_source)


def test_additional_namespace_location_rejected(execution_source, tmp_path):
    *_, modules = execution_source
    modules["scripts"].__path__.append(str(tmp_path / "stale/scripts"))
    with pytest.raises(ValueError, match="namespace path"):
        bind_execution(execution_source)


def test_bare_stale_postfit_helper_rejected(execution_source, tmp_path):
    *_, modules = execution_source
    stale = tmp_path / "condition_relevance_transfer.py"
    stale.write_text("# stale scoring helper\n", encoding="utf-8")
    modules["condition_relevance_transfer"] = SimpleNamespace(__file__=str(stale))
    with pytest.raises(ValueError, match="imported module origin"):
        bind_execution(execution_source)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def files(tmp_path, monkeypatch):
    ce, aux = pair()
    runs = {"ce": tmp_path / "ce", "relevance": tmp_path / "relevance"}
    source = tmp_path / "source"
    corpus = tmp_path / "corpus"
    model = tmp_path / "base" / "model"
    for p in (*runs.values(), source, corpus, model):
        p.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"synthetic weights; not a model")
    base_manifest = model.parent / "download.json"
    write_json(
        base_manifest,
        {
            "revision": contract.REVISION,
            "hashes": {"model.safetensors": sha(model / "model.safetensors")},
        },
    )
    development = [
        "binding_train",
        "numeric_train",
        "legacy_train",
        "validation",
        "old_validation",
        "legacy_validation",
    ]
    for name in development:
        (corpus / (name + ".jsonl")).write_bytes(b'{"split":"fixture"}\n')
    write_json(corpus / "selection-schema.json", {"fixture": 1})
    (corpus / "transfer.jsonl").write_bytes(b'{"split":"transfer","synthetic_fixture":true}\n')
    manifest = {
        "status": "PASS_FROZEN_CONDITION_RELEVANCE",
        "sealed_audit_loaded": False,
        "new_transfer_model_evaluated": False,
        "old_transfer_payload_opened": False,
        "old_state_partitions_opened": False,
        "calibration_loaded_for_fit": False,
        "artifacts": {p.name: sha(p) for p in corpus.iterdir()},
    }
    write_json(corpus / "result.json", manifest)
    monkeypatch.setattr(contract, "CORPUS_SHA", sha(corpus / "result.json"))
    config = ce["config"]
    config.update(
        corpus=str(corpus),
        model=str(model),
        corpus_sha256=contract.CORPUS_SHA,
        model_manifest_sha256=sha(base_manifest),
    )
    config_path = source / "configs/pretrained_reuse/condition_relevance_fit_v1.json"
    write_json(config_path, config)
    runner_path = source / "scripts/train_condition_relevance.py"
    runner_path.parent.mkdir()
    runner_path.write_text("# synthetic fixture runner\n", encoding="utf-8")
    monkeypatch.setattr(contract, "CONFIG_SHA", sha(config_path))
    input_pins = {
        str(corpus / "result.json"): contract.CORPUS_SHA,
        str(base_manifest): sha(base_manifest),
        str(model / "model.safetensors"): sha(model / "model.safetensors"),
    }
    input_pins.update(
        {str(corpus / (n + ".jsonl")): sha(corpus / (n + ".jsonl")) for n in development}
    )
    input_pins[str(corpus / "selection-schema.json")] = sha(corpus / "selection-schema.json")
    for arm, run in [("ce", ce), ("relevance", aux)]:
        run.update(
            config=config,
            config_sha256=contract.CONFIG_SHA,
            input_pins=input_pins,
            code_pins={str(config_path): sha(config_path), str(runner_path): sha(runner_path)},
        )
        run["command"] = [
            str(source / "scripts/train_condition_relevance.py"),
            "--arm",
            arm,
            "--output",
            str(runs[arm]),
        ]
    effective = effective_pair(ce, aux)
    for arm, run in [("ce", ce), ("relevance", aux)]:
        for name in run["artifacts"]:
            (runs[arm] / name).write_bytes(name.encode())
        (runs[arm] / "best-delta.pt").write_bytes(
            (runs[arm] / f"step-{run['best_step']}-delta.pt").read_bytes()
        )
        write_json(runs[arm] / "effective-config.json", effective[arm])
        run["artifacts"] = {p.name: sha(p) for p in runs[arm].iterdir()}
        run["selected_sha256"] = run["artifacts"]["best-delta.pt"]
        write_json(runs[arm] / "result.json", run)
    return {
        "runs": runs,
        "hashes": {a: sha(p / "result.json") for a, p in runs.items()},
        "source": source,
        "corpus": corpus,
        "model": model,
        "metadata": {"ce": ce, "relevance": aux},
    }


def io():
    from scripts import condition_relevance_postfit_io

    return condition_relevance_postfit_io


def inspect(f):
    return io().verify_pair_files(f["runs"], f["hashes"], f["source"])


def update_run(f, arm):
    path = f["runs"][arm] / "result.json"
    write_json(path, f["metadata"][arm])
    f["hashes"][arm] = sha(path)


def test_real_files_bind_freeze_and_transfer_bytes(files, tmp_path):
    checked = inspect(files)
    destination = tmp_path / "frozen.json"
    frozen_sha = io().write_freeze(checked, destination)
    raw = io().read_frozen_transfer(
        destination, frozen_sha, files["runs"], files["hashes"], files["source"]
    )
    assert json.loads(raw)["synthetic_fixture"] is True
    assert (
        checked["frozen"]["effective_config_hashes"]["ce"]
        != checked["frozen"]["effective_config_hashes"]["relevance"]
    )
    assert sha(destination) == frozen_sha


@pytest.mark.parametrize("failure", ["running", "bad_result_hash"])
def test_rejects_before_effective_or_protected_payload_io(files, monkeypatch, failure):
    if failure == "running":
        files["metadata"]["relevance"]["status"] = "RUNNING"
        update_run(files, "relevance")
    else:
        files["hashes"]["relevance"] = "0" * 64
    original = Path.read_bytes
    seen = []

    def read(path):
        seen.append(path)
        assert path.name == "result.json" and path.parent in files["runs"].values()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError):
        inspect(files)
    assert seen


@pytest.mark.parametrize("name", ["effective-config.json", "step-1024-delta.pt"])
def test_modified_run_bytes_rejected(files, name):
    (files["runs"]["relevance"] / name).write_bytes(b"changed")
    with pytest.raises(ValueError):
        inspect(files)


def test_modified_source_bytes_rejected(files):
    path = next(iter(files["metadata"]["ce"]["code_pins"]))
    Path(path).write_bytes(b"changed")
    with pytest.raises(ValueError):
        inspect(files)


@pytest.mark.parametrize("name", ["../outside.bin", "transfer.jsonl", "sealed_audit.jsonl"])
def test_unregistered_run_artifact_path_rejected(files, name):
    for arm in files["runs"]:
        files["metadata"][arm]["artifacts"][name] = "a" * 64
        update_run(files, arm)
    with pytest.raises(ValueError):
        inspect(files)


def test_protected_partition_cannot_be_added_to_input_pins(files):
    for arm in files["runs"]:
        files["metadata"][arm]["input_pins"][str(files["corpus"] / "transfer.jsonl")] = "a" * 64
        effective = effective_pair(files["metadata"]["ce"], files["metadata"]["relevance"])[arm]
        path = files["runs"][arm] / "effective-config.json"
        write_json(path, effective)
        files["metadata"][arm]["artifacts"][path.name] = sha(path)
        update_run(files, arm)
    with pytest.raises(ValueError):
        inspect(files)


def test_changed_frozen_roles_rejected_before_transfer_read(files, tmp_path, monkeypatch):
    checked = inspect(files)
    destination = tmp_path / "frozen.json"
    checked["frozen"]["roles"]["ce_endpoint"]["step"] = 512
    digest = io().write_freeze(checked, destination)
    original = Path.read_bytes

    def read(path):
        assert path.name != "transfer.jsonl", "changed freeze reached protected transfer"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError):
        io().read_frozen_transfer(
            destination, digest, files["runs"], files["hashes"], files["source"]
        )


def test_changed_transfer_bytes_rejected(files, tmp_path):
    checked = inspect(files)
    destination = tmp_path / "frozen.json"
    digest = io().write_freeze(checked, destination)
    (files["corpus"] / "transfer.jsonl").write_bytes(b"not even JSON")
    with pytest.raises(ValueError):
        io().read_frozen_transfer(
            destination, digest, files["runs"], files["hashes"], files["source"]
        )


def test_freeze_never_overwrites_or_writes_into_protected_paths(files, tmp_path):
    checked = inspect(files)
    destination = tmp_path / "frozen.json"
    io().write_freeze(checked, destination)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        io().write_freeze(checked, destination)
    assert destination.read_bytes() == before
    for root in [*files["runs"].values(), files["source"], files["corpus"], files["model"]]:
        with pytest.raises(ValueError):
            io().write_freeze(checked, root / "forbidden.json")


@pytest.mark.parametrize("filename", ["result.json", "effective-config.json", "step-1024-delta.pt"])
def test_redirected_run_child_rejected_before_open(files, monkeypatch, filename):
    child = files["runs"]["relevance"] / filename
    protected = files["corpus"] / "transfer.jsonl"
    original_resolve, original_open = Path.resolve, Path.open

    def resolve(path, *args, **kwargs):
        return protected if path == child else original_resolve(path, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        assert path not in (child, protected), "redirected file opened before path rejection"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(ValueError):
        inspect(files)


@pytest.mark.parametrize("change", ["copied_runner", "missing_runner_pin"])
def test_executed_runner_must_be_the_pinned_source(files, change):
    for arm in files["runs"]:
        run = files["metadata"][arm]
        if change == "copied_runner":
            run["command"][0] = str(
                files["source"].parent / "copy/scripts/train_condition_relevance.py"
            )
        else:
            run["code_pins"].pop(
                str(files["source"] / "scripts/train_condition_relevance.py"), None
            )
    values = effective_pair(files["metadata"]["ce"], files["metadata"]["relevance"])
    for arm in files["runs"]:
        path = files["runs"][arm] / "effective-config.json"
        write_json(path, values[arm])
        files["metadata"][arm]["artifacts"][path.name] = sha(path)
        update_run(files, arm)
    with pytest.raises(ValueError):
        inspect(files)
