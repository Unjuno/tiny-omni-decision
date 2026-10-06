from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.audio_evaluation import (  # noqa: E402
    build_hard_negative_example,
    hard_negative_score,
)
from tiny_omni_decision.corpus import macro_metrics  # noqa: E402
from tiny_omni_decision.dataset import iter_local_rows  # noqa: E402
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import BaseModelManifest, DecisionExample  # noqa: E402
from tiny_omni_decision.trainer import (  # noqa: E402
    _evaluate,
    _load_pretrained_base,
    resolve_media_root,
)
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    processor_inputs_for_example,
)

RUN = ROOT / "artifacts/tiny-omni-decision-teacher-v2/clean-dev-v2-seed17-2048"
VALIDATION = ROOT / "data/processed/teacher-quality-next/clean-dev-v2/validation.jsonl"
CONFIG = ROOT / "configs/decision/teacher_quality_clean_dev_v2.yaml"
BASE_MANIFEST = ROOT / "manifests/base-model.example.yaml"
OUTPUT = ROOT / "artifacts/teacher-quality-next/evaluations/librispeech-hard-negative-step768-v1"
PRIOR_EVALUATION = OUTPUT
PINNED_RESUME_SNAPSHOT = RUN / "checkpoints/resume-step-000896-ce5b44dfa4914c019c28ac8a11c40b2a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare() -> Path:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {OUTPUT}")
    selector_path = RUN / "validation-selector.json"
    predictions_path = RUN / "validation-step-768-predictions.jsonl"
    metrics_path = RUN / "validation-step-768.json"
    prior_manifest_path = PRIOR_EVALUATION / "manifest.json"
    for path in (VALIDATION, selector_path, predictions_path, metrics_path, prior_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    best = prior_manifest.get("best_checkpoint_metadata", {})
    if prior_manifest.get("same_model_checkpoint_step") != 768 or best.get("step") != 768:
        raise ValueError("the paired hard-negative evaluation is pinned to best step 768")
    if prior_manifest.get("source_validation_sha256") != sha256_file(VALIDATION):
        raise ValueError("source validation differs from the pinned step-768 evaluation")
    snapshot_state = json.loads(
        (PINNED_RESUME_SNAPSHOT / "trainer-state.json").read_text(encoding="utf-8")
    )
    if snapshot_state.get("global_step") != 896 or snapshot_state.get("best_step") != 768:
        raise ValueError("pinned resume snapshot no longer retains the step-768 adapter")
    if sha256_file(PINNED_RESUME_SNAPSHOT / "best/adapter_model.safetensors") != best.get(
        "adapter_sha256"
    ):
        raise ValueError("step-768 adapter hash differs from the prior evaluation manifest")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    selector_ids = set(selector["ids"])
    all_rows = list(iter_local_rows(VALIDATION))
    validation_audio = {
        row["id"]: row
        for row in all_rows
        if row.get("modality") == "audio"
        and row.get("source") == "openslr/LibriSpeech"
        and row.get("split") == "validation"
    }
    selected = [
        validation_audio[sample_id]
        for sample_id in selector["ids"]
        if sample_id in validation_audio
    ]
    if len(selected) != 512 or len({row["id"] for row in selected}) != 512:
        raise ValueError("step-768 selector must contain exactly 512 unique LibriSpeech audio rows")
    transcript_bank = [row for row in validation_audio.values() if row["id"] not in selector_ids]
    if len(transcript_bank) < 3:
        raise ValueError("validation transcript bank is too small after selector isolation")

    hard_rows: list[dict[str, object]] = []
    distractor_audit: list[dict[str, object]] = []
    for row in selected:
        hard_row, distractors = build_hard_negative_example(row, transcript_bank)
        hard_rows.append(hard_row)
        distractor_audit.append({"sample_id": row["id"], "distractors": distractors})
    examples = [DecisionExample.model_validate(row) for row in hard_rows]
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("hard-negative example IDs are not unique")
    if any(
        len(example.options) != 4 or example.target not in example.options for example in examples
    ):
        raise ValueError("hard-negative rows must preserve four options and the correct answer")

    baseline_predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_audio = [row for row in baseline_predictions if row.get("modality") == "audio"]
    baseline_ids = {row["sample_id"] for row in baseline_audio}
    selected_ids = {row["id"] for row in selected}
    if baseline_ids != selected_ids or len(baseline_audio) != 512:
        raise ValueError("step-768 baseline predictions do not match the 512 hard-negative IDs")
    original_examples = [DecisionExample.model_validate(row) for row in selected]

    original_distractor_scores = [
        hard_negative_score(str(row["target"]), str(option))[0]
        for row in selected
        for option in row["options"]
        if option != row["target"]
    ]
    selected_hard_scores = [
        float(distractor["weighted_similarity"])
        for row in distractor_audit
        for distractor in row["distractors"]
    ]
    hard_best_by_row = [
        max(float(distractor["weighted_similarity"]) for distractor in row["distractors"])
        for row in distractor_audit
    ]
    original_best_by_row = [
        max(
            hard_negative_score(str(row["target"]), str(option))[0]
            for option in row["options"]
            if option != row["target"]
        )
        for row in selected
    ]

    OUTPUT.mkdir(parents=True, exist_ok=False)
    corpus_path = OUTPUT / "hard-negative-validation.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in hard_rows),
        encoding="utf-8",
    )
    original_corpus_path = OUTPUT / "original-validation.jsonl"
    original_corpus_path.write_text(
        "".join(
            json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for example in original_examples
        ),
        encoding="utf-8",
    )
    (OUTPUT / "baseline-original-audio-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in baseline_audio),
        encoding="utf-8",
    )
    (OUTPUT / "distractor-audit.json").write_text(
        json.dumps(distractor_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["validation"][
        "modality:audio"
    ]
    manifest = {
        "evaluation_id": OUTPUT.name,
        "status": "prepared; matched paired inference pending",
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "similarity_module_sha256": sha256_file(
            ROOT / "src/tiny_omni_decision/audio_evaluation.py"
        ),
        "baseline_condition": "original 4-choice LibriSpeech validation distractors",
        "hard_negative_condition": (
            "3 highest-scoring distractors from non-selector LibriSpeech validation transcripts"
        ),
        "hard_negative_score": (
            "0.50 Soundex-sequence similarity + 0.30 word-set Jaccard + 0.20 character-length ratio"
        ),
        "phonetic_method_limit": (
            "Soundex is a deterministic English phonetic proxy, not a "
            "pronunciation lexicon or human ambiguity annotation."
        ),
        "same_audio_ids": len(selected_ids),
        "same_model_checkpoint_step": 768,
        "best_checkpoint_metadata": best,
        "baseline_metrics": baseline_metrics,
        "source_validation_sha256": sha256_file(VALIDATION),
        "training_config_sha256": sha256_file(CONFIG),
        "base_model_manifest_sha256": sha256_file(BASE_MANIFEST),
        "selector_sha256": sha256_file(selector_path),
        "baseline_metrics_sha256": sha256_file(metrics_path),
        "baseline_predictions_sha256": sha256_file(predictions_path),
        "hard_corpus_sha256": sha256_file(corpus_path),
        "original_corpus_sha256": sha256_file(original_corpus_path),
        "hard_sample_id_order_sha256": hashlib.sha256(
            "".join(f"{example.id}\n" for example in examples).encode()
        ).hexdigest(),
        "selector_audio_sample_ids": sorted(selected_ids),
        "distractor_source_ids_excluded_from_pool": sorted(selector_ids),
        "distractor_count_by_candidate_source": dict(
            Counter(row["source"] for row in transcript_bank)
        ),
        "similarity_proxy_summary": {
            "original_wrong_choice_mean": sum(original_distractor_scores)
            / len(original_distractor_scores),
            "hard_wrong_choice_mean": sum(selected_hard_scores) / len(selected_hard_scores),
            "original_best_wrong_choice_mean": sum(original_best_by_row)
            / len(original_best_by_row),
            "hard_best_wrong_choice_mean": sum(hard_best_by_row) / len(hard_best_by_row),
            "rows_hard_best_exceeds_original_best": sum(
                hard > original
                for hard, original in zip(hard_best_by_row, original_best_by_row, strict=True)
            ),
        },
        "sealed_audit_loaded": False,
        "training_process_or_checkpoint_modified": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "manifest": manifest}, indent=2))
    return corpus_path


def evaluate() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    corpus_path = OUTPUT / "hard-negative-validation.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    if not corpus_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run the script once to prepare the hard-negative set")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("same_model_checkpoint_step") != 768:
        raise ValueError("hard-negative set is not pinned to the paired step-768 checkpoint")
    if sha256_file(corpus_path) != manifest.get("hard_corpus_sha256"):
        raise ValueError("hard-negative corpus hash differs from its manifest")
    if (OUTPUT / "hard-negative-metrics.json").exists():
        raise FileExistsError("refusing to overwrite the hard-negative evaluation result")

    examples = [DecisionExample.model_validate(row) for row in iter_local_rows(corpus_path)]
    original_corpus_path = OUTPUT / "original-validation.jsonl"
    if sha256_file(original_corpus_path) != manifest.get("original_corpus_sha256"):
        raise ValueError("original validation corpus hash differs from its manifest")
    original_examples = [
        DecisionExample.model_validate(row) for row in iter_local_rows(original_corpus_path)
    ]
    if {example.id for example in original_examples} != {
        example.id.removesuffix(":hard-neg-v1") for example in examples
    }:
        raise ValueError("original and hard-negative evaluation IDs are not paired")
    config = decision_training_config(load_structured_file(CONFIG))
    base_manifest = BaseModelManifest.model_validate(load_structured_file(BASE_MANIFEST))
    processor = AutoProcessor.from_pretrained(
        base_manifest.processor_repo_id or base_manifest.repo_id,
        revision=base_manifest.processor_revision,
        local_files_only=True,
    )
    configured_max_sequence_length = config.max_sequence_length
    data_root = resolve_media_root(VALIDATION, ROOT / "data")
    observed_lengths = [
        int(
            processor_inputs_for_example(
                processor,
                example,
                data_root=data_root,
                video_num_frames=config.video_num_frames,
            )[0]["input_ids"].shape[-1]
        )
        for example in [*original_examples, *examples]
    ]
    effective_max_sequence_length = max(configured_max_sequence_length, max(observed_lengths))
    config = config.model_copy(update={"max_sequence_length": effective_max_sequence_length})
    if not torch.cuda.is_available():
        raise RuntimeError("hard-negative model evaluation requires the existing local CUDA GPU")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    base = _load_pretrained_base(
        AutoModelForMultimodalLM.from_pretrained,
        base_manifest.repo_id,
        revision=base_manifest.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    snapshot_state = json.loads(
        (PINNED_RESUME_SNAPSHOT / "trainer-state.json").read_text(encoding="utf-8")
    )
    if snapshot_state.get("global_step") != 896 or snapshot_state.get("best_step") != 768:
        raise ValueError("resume snapshot does not retain the pinned step-768 best adapter")
    snapshot_metadata = json.loads(
        (PINNED_RESUME_SNAPSHOT / "best-checkpoint.json").read_text(encoding="utf-8")
    )
    if snapshot_metadata != manifest["best_checkpoint_metadata"]:
        raise ValueError("step-896 resume snapshot best metadata differs from the baseline model")
    adapter_path = PINNED_RESUME_SNAPSHOT / "best"
    if (
        sha256_file(adapter_path / "adapter_model.safetensors")
        != manifest["best_checkpoint_metadata"]["adapter_sha256"]
    ):
        raise ValueError("read-only best adapter hash differs from the paired checkpoint metadata")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).eval()
    def progress(completed: int, total: int, *, condition: str) -> None:
        if completed % 50 == 0 or completed == total:
            print(f"{condition} evaluation {completed}/{total}", flush=True)

    baseline_started = time.monotonic()
    baseline_metrics, baseline_predictions = _evaluate(
        model,
        processor,
        original_examples,
        data_root=data_root,
        config=config,
        progress_callback=lambda completed, total: progress(
            completed, total, condition="Original-choice"
        ),
    )
    baseline_seconds = time.monotonic() - baseline_started
    baseline_metrics["macro"] = macro_metrics(baseline_metrics)
    baseline_metrics["configured_max_sequence_length"] = configured_max_sequence_length
    baseline_metrics["effective_max_sequence_length"] = effective_max_sequence_length

    hard_started = time.monotonic()
    metrics, predictions = _evaluate(
        model,
        processor,
        examples,
        data_root=data_root,
        config=config,
        progress_callback=lambda completed, total: progress(
            completed, total, condition="Hard-negative"
        ),
    )
    hard_seconds = time.monotonic() - hard_started
    metrics["macro"] = macro_metrics(metrics)
    metrics["configured_max_sequence_length"] = configured_max_sequence_length
    metrics["effective_max_sequence_length"] = effective_max_sequence_length
    hard_audio_metrics = metrics["modality:audio"]
    comparison = {
        "paired_examples": len(examples),
        "checkpoint_step": 768,
        "checkpoint_adapter_sha256": manifest["best_checkpoint_metadata"]["adapter_sha256"],
        "original_distractors": baseline_metrics["modality:audio"],
        "hard_negative_distractors": hard_audio_metrics,
        "delta_hard_minus_original": {
            key: hard_audio_metrics[key] - baseline_metrics["modality:audio"][key]
            for key in ("accuracy", "nll", "brier", "ece")
        },
        "original_evaluation_seconds": baseline_seconds,
        "hard_evaluation_seconds": hard_seconds,
        "configured_max_sequence_length": configured_max_sequence_length,
        "effective_max_sequence_length": effective_max_sequence_length,
        "hard_corpus_sha256": manifest["hard_corpus_sha256"],
        "original_training_predictions_retained": True,
        "training_process_or_checkpoint_modified": False,
    }
    (OUTPUT / "hard-negative-metrics.json").write_text(
        json.dumps({"metrics": metrics, "comparison": comparison}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "baseline-matched-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in baseline_predictions),
        encoding="utf-8",
    )
    (OUTPUT / "hard-negative-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    manifest["status"] = "evaluated"
    manifest["configured_max_sequence_length"] = configured_max_sequence_length
    manifest["effective_max_sequence_length"] = effective_max_sequence_length
    manifest["matched_baseline_metrics"] = baseline_metrics["modality:audio"]
    manifest["matched_baseline_metrics_sha256"] = hashlib.sha256(
        json.dumps(baseline_metrics, sort_keys=True).encode()
    ).hexdigest()
    manifest["hard_metrics_sha256"] = sha256_file(OUTPUT / "hard-negative-metrics.json")
    manifest["matched_baseline_predictions_sha256"] = sha256_file(
        OUTPUT / "baseline-matched-predictions.jsonl"
    )
    manifest["hard_predictions_sha256"] = sha256_file(OUTPUT / "hard-negative-predictions.jsonl")
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser(
        description="Paired LibriSpeech hard-negative audio evaluation."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if not args.evaluate_only:
        prepare()
    if not args.prepare_only:
        evaluate()


if __name__ == "__main__":
    main()
