"""Frozen two-model inference screen, with explicit preparation and fresh replay."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.build_condition_binding_corpus import lines
from scripts.condition_binding_corpus import identity
from scripts.evaluate_semantic_state_bridge import read, sha, verify, write
from scripts.factorized_condition import (
    FIELDS,
    combine,
    control_rows,
    diagnostic_summary,
    make_rows,
    order_control,
    render,
    replay_equal,
    research_failures,
    select_states,
)
from scripts.semantic_state_diagnostic import log_probabilities
from scripts.summarize_condition_binding import score_difference

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
R1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
CODE = [
    "scripts/factorized_condition.py",
    "scripts/run_factorized_condition.py",
    "tests/test_factorized_condition.py",
    "scripts/condition_binding_corpus.py",
    "scripts/build_condition_binding_corpus.py",
    "scripts/condition_binding_training.py",
    "scripts/semantic_state_diagnostic.py",
    "scripts/summarize_condition_binding.py",
    "scripts/train_text_crossencoder.py",
    "scripts/evaluate_semantic_state_bridge.py",
    "src/tiny_omni_decision/dataset.py",
    "docs/pretrained_reuse/R2_FACTORIZED_REPRESENTATION_PROTOCOL_20261015.md",
]
MODEL_INFO = {
    "minilm_kl512": dict(
        path=str(R1 / "crossencoder-minilm-l6-audit-01/model"),
        parameters=22713601,
        revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
        delta=str(ROOT / "condition-binding-retention-fit-01/best-delta.pt"),
        delta_sha256="c87ddf2197e276814e4941efb7ad7431d416316eb786ea59b8331925ca16ab02",
    ),
    "deberta_xsmall": dict(
        path=str(R1 / "deberta-xsmall-zeroshot-audit-01/model"),
        parameters=70830722,
        revision="262ae02f29173eec1c250f90804dc7edc677dcff",
    ),
}


def code_pins():
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *CODE], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", *CODE], check=True, capture_output=True
    )
    return {str(Path(p).resolve()): sha(p) for p in CODE}


def prepare(output):
    sources = [
        "clevr4-full-coverage-seed17/train-labels.jsonl",
        "semantic-state-bridge-01/states.json",
        "condition-binding-corpus-01/state-partitions.json",
        "binding-independent-transfer-corpus-01/states.json",
        "binding-language-corpus-01/states.json",
        "condition-contrast-corpus-01/state-partitions.json",
    ]
    pins, excluded = {}, []
    for name in sources:
        path = ROOT / name
        pins[str(path)] = sha(path)
        data = lines(path) if path.suffix == ".jsonl" else read(path)
        excluded += sum(data.values(), []) if isinstance(data, dict) else data
    for name, info in MODEL_INFO.items():
        folder = Path(info["path"])
        manifest = folder.parent / (
            "result-retry.json" if name == "minilm_kl512" else "download.json"
        )
        pins[str(manifest)] = sha(manifest)
        m = read(manifest)
        hashes = {p: x["sha256"] for p, x in m["files"].items()} if "files" in m else m["hashes"]
        pins.update({str(folder / p): h for p, h in hashes.items()})
        if info.get("delta"):
            pins[info["delta"]] = info["delta_sha256"]
    verify(pins)
    states = select_states(excluded)
    rows = make_rows(states)
    if len(rows) != 1024 or {identity(s) for s in states} & {identity(s) for s in excluded}:
        raise ValueError("cohort integrity failure")
    code = code_pins()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "states.json", states)
    write(output / "rows.json", rows)
    write(output / "excluded-identities.json", sorted({identity(s) for s in excluded}))
    result = dict(
        status="PASS_FROZEN_FACTORIZATION_CORPUS",
        code_pins=code,
        input_pins=pins,
        models=MODEL_INFO,
        states=32,
        questions=1024,
        candidate_count=40,
        excluded_unique_states=len({identity(s) for s in excluded}),
        state_overlap=0,
        native_assets=0,
        training=False,
        sealed_audit_loaded=False,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        artifacts={p.name: sha(p) for p in output.iterdir()},
    )
    write(output / "result.json", result)
    print(json.dumps(dict(output=str(output), sha256=sha(output / "result.json"))), flush=True)


def evaluate(args):
    tick = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        command=sys.argv,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        training=False,
        sealed_audit_loaded=False,
        product_complete=False,
        cloud_cost=0,
        model=args.model,
        smoke=args.smoke,
    )
    write(args.output / "result.json", result)
    try:
        verify({str(args.corpus / "result.json"): args.corpus_sha256})
        corpus = read(args.corpus / "result.json")
        code = code_pins()
        if corpus["status"] != "PASS_FROZEN_FACTORIZATION_CORPUS" or corpus["code_pins"] != code:
            raise ValueError("frozen corpus/code mismatch")
        verify(corpus["input_pins"])
        verify({str(args.corpus / p): h for p, h in corpus["artifacts"].items()})
        if corpus["models"] != MODEL_INFO:
            raise ValueError("model contract mismatch")
        rows = read(args.corpus / "rows.json")
        if args.smoke:
            rows = rows[:8]  # two whole states, in a separate process/output
        if args.replay_reference:
            verify({str(args.replay_reference / "result.json"): args.replay_sha256})
            reference = read(args.replay_reference / "result.json")
            if (
                reference["status"] != "PASS_FACTORIZATION_INFERENCE"
                or reference["smoke"]
                or reference["model"] != args.model
                or reference["code_pins"] != code
                or reference["corpus_sha256"] != args.corpus_sha256
            ):
                raise ValueError("completed exact-contract primary required")
            verify({str(args.replay_reference / p): h for p, h in reference["artifacts"].items()})
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import psutil
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from scripts.train_text_crossencoder import frozen_hash

        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        info = MODEL_INFO[args.model]
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            info["path"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            output_loading_info=True,
        )
        if any(loading.values()):
            raise ValueError(f"checkpoint load mismatch: {loading}")
        if "delta" in info:
            delta = torch.load(info["delta"], weights_only=True, map_location="cpu")
            expected = {
                n
                for n, _ in model.named_parameters()
                if n.startswith(("bert.encoder.layer.", "classifier."))
            }
            if set(delta) != expected:
                raise ValueError("delta scope mismatch")
            loaded = model.load_state_dict(delta, strict=False)
            if (
                loaded.unexpected_keys
                or set(loaded.missing_keys) != set(model.state_dict()) - expected
            ):
                raise ValueError("delta loading mismatch")
        elif model.config.label2id.get("entailment") != 0:
            raise ValueError("native entailment label changed")
        model = model.cuda().eval().requires_grad_(False)
        count = sum(p.numel() for p in model.parameters())
        if count != info["parameters"]:
            raise ValueError("parameter count mismatch")
        before = frozen_hash(model)
        tokenizer = AutoTokenizer.from_pretrained(
            info["path"], local_files_only=True, trust_remote_code=False
        )
        torch.cuda.reset_peak_memory_stats()
        accounting = []
        environment = dict(
            python=sys.version,
            torch=torch.__version__,
            transformers=transformers.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            dtype="float32",
            attention=model.config._attn_implementation,
            batch_size=32,
            threads=4,
        )
        write(
            args.output / "effective-config.json",
            dict(
                model=info,
                environment=environment,
                corpus_sha256=args.corpus_sha256,
                code_pins=code,
                input_pins=corpus["input_pins"],
                frozen_sha256=before,
                parameters=count,
                smoke=args.smoke,
            ),
        )

        def guard():
            if (
                time.perf_counter() - tick > 900
                or torch.cuda.max_memory_allocated() > 8 * 2**30
                or psutil.Process().memory_info().peak_wset > 6 * 2**30
                or shutil.disk_usage(args.output).free < 2 * 2**30
                or sum(p.stat().st_size for p in args.output.iterdir()) > 512 * 2**20
            ):
                raise RuntimeError("resource guard: incomplete experiment")

        def score(pairs, label):
            start = time.perf_counter()
            unique = list(dict.fromkeys(pairs))
            table = {}
            max_tokens = 0
            with torch.inference_mode():
                for offset in range(0, len(unique), 32):
                    guard()
                    group = unique[offset : offset + 32]
                    batch = tokenizer(
                        [p[0] for p in group],
                        [p[1] for p in group],
                        truncation=False,
                        padding=True,
                        return_tensors="pt",
                    )
                    max_tokens = max(max_tokens, batch["input_ids"].shape[1])
                    if max_tokens > 512:
                        raise ValueError("truncation prohibited")
                    logits = model(**batch.to("cuda")).logits[:, 0].float().cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError("nonfinite model logits")
                    table.update(zip(group, logits.tolist(), strict=True))
            guard()
            accounting.append(
                dict(
                    label=label,
                    requested=len(pairs),
                    unique=len(unique),
                    max_tokens=max_tokens,
                    seconds=time.perf_counter() - start,
                )
            )
            return np.array([table[p] for p in pairs], dtype=np.float64)

        summaries, arrays, controls = {}, {}, {}
        for control in ("primary", "reverse", "donor", "no_question"):
            cohort = control_rows(rows, control)
            rendered = [render(r) for r in cohort]
            pieces = {
                key: score([p for r in rendered for p in r[key]], f"{control}-{key}").reshape(
                    len(rows), width
                )
                for key, width in (("direct", 40), ("routing", 4), ("values", 40))
            }
            direct = pieces["direct"]
            factored = np.stack(
                [
                    combine(q, v, FIELDS, r["option_ids"])
                    for q, v, r in zip(pieces["routing"], pieces["values"], cohort, strict=True)
                ]
            )
            for key, data in dict(**pieces, factorized=factored).items():
                np.save(args.output / f"{control}-{key}.npy", data, allow_pickle=False)
            summaries[control] = {}
            arrays[control] = dict(direct=direct, factorized=factored)
            for name, values in arrays[control].items():
                summaries[control][name] = diagnostic_summary(cohort, values)
                probability = np.exp(log_probabilities(values))
                write(
                    args.output / f"{control}-{name}-predictions.json",
                    [
                        dict(
                            id=r["id"],
                            target_id=r["target_id"],
                            option_ids=r["option_ids"],
                            prediction_id=r["option_ids"][int(p.argmax())],
                            probabilities=p.tolist(),
                        )
                        for r, p in zip(cohort, probability, strict=True)
                    ],
                )
                if control == "donor":
                    summaries[control][name]["against_original_targets"] = diagnostic_summary(
                        rows, values
                    )
                if control == "reverse":
                    original = np.exp(log_probabilities(arrays["primary"][name]))
                    controls[name] = order_control(
                        original,
                        probability,
                        [r["option_ids"] for r in rows],
                        [r["option_ids"] for r in cohort],
                    )
            print(
                json.dumps(
                    dict(
                        model=args.model,
                        condition=control,
                        accuracy={
                            k: v["overall"]["accuracy"] for k, v in summaries[control].items()
                        },
                    )
                ),
                flush=True,
            )
            guard()
        paired = score_difference(
            rows, arrays["primary"]["direct"], arrays["primary"]["factorized"]
        )
        failures = research_failures(
            summaries["primary"]["direct"], summaries["primary"]["factorized"], paired
        )
        if frozen_hash(model) != before:
            raise ValueError("frozen parameters changed")
        verify(corpus["input_pins"])
        verify(code)
        write(
            args.output / "metrics.json",
            dict(
                conditions=summaries,
                order_controls=controls,
                paired=paired,
                research_failures=failures,
            ),
        )
        replay = []
        if args.replay_reference:
            for path in args.output.iterdir():
                if path.suffix == ".npy":
                    a = np.load(path, allow_pickle=False)
                    b = np.load(args.replay_reference / path.name, allow_pickle=False)
                    if not np.allclose(a, b, rtol=0, atol=1e-6):
                        raise ValueError(f"fresh replay mismatch: {path.name}")
                    replay.append(dict(name=path.name, max_delta=float(abs(a - b).max())))
            payloads = [args.output / "metrics.json", *args.output.glob("*-predictions.json")]
            for path in payloads:
                if not replay_equal(read(path), read(args.replay_reference / path.name)):
                    raise ValueError(f"full discrete/numeric replay mismatch: {path.name}")
                replay.append(dict(name=path.name, discrete_exact_numeric_atol=1e-6))
        guard()
        result.update(
            status="PASS_FACTORIZATION_INFERENCE",
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - tick,
            questions=len(rows),
            parameters=count,
            environment=environment,
            code_pins=code,
            corpus_sha256=args.corpus_sha256,
            frozen_weights_unchanged=True,
            input_pins=corpus["input_pins"],
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_host_bytes=psutil.Process().memory_info().peak_wset,
            forward_accounting=accounting,
            replay=replay,
            metrics=summaries,
            research_failures=failures,
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.name != "result.json"},
        )
        write(args.output / "result.json", result)
        guard()
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    seconds=result["seconds"],
                    research_failures=failures,
                    sha256=sha(args.output / "result.json"),
                )
            ),
            flush=True,
        )
    except Exception as error:
        result.update(
            status="FAIL_INCOMPLETE",
            error=repr(error),
            traceback=traceback.format_exc(),
            seconds=time.perf_counter() - tick,
        )
        write(args.output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-sha256")
    parser.add_argument("--model", choices=list(MODEL_INFO))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if args.prepare:
        if args.corpus or args.model or args.smoke or args.replay_reference:
            parser.error("preparation takes only a new output directory")
        prepare(args.output)
    else:
        if (
            not args.corpus
            or not args.corpus_sha256
            or not args.model
            or bool(args.replay_reference) != bool(args.replay_sha256)
            or args.smoke
            and args.replay_reference
        ):
            parser.error("complete frozen corpus/model and paired replay arguments required")
        evaluate(args)


if __name__ == "__main__":
    main()
