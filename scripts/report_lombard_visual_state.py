"""Publish hash-bound primary/replay evidence; no model or metric selection."""

import hashlib
import json
from pathlib import Path

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
DOCS = Path("docs/pretrained_reuse")
NAMES = ["lombard-visual-state-diagnosis-01", "lombard-visual-state-replay-01"]
HASHES = [
    "25076e30d398defcfcb5c7a5123c27aed371703354c3902bdd927e898d56897f",
    "c1ac0b1d699d623048388dfe61f2166aead7608f9449a12f198574b5483632a0",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def verify(pins):
    if not pins or any(sha(p) != h for p, h in pins.items()):
        raise ValueError("publication source/report/control hash differs")


def main():
    pins = {str(ROOT / n / "result.json"): h for n, h in zip(NAMES, HASHES, strict=True)}
    verify(pins)
    reports = [json.loads((ROOT / n / "result.json").read_text()) for n in NAMES]
    first, replay = reports
    if (
        any(r["status"] != "PASS_FIXED_VISUAL_STATE_DIAGNOSIS" for r in reports)
        or replay["fresh_process_replay"] is not True
        or replay["arms"] != first["arms"]
    ):
        raise ValueError("complete bitwise fresh replay required")
    for name, r in zip(NAMES, reports, strict=True):
        pins.update(r["validated_hashes"])
        pins.update({str(ROOT / name / f): h for f, h in r["artifact_hashes"].items()})
    pins[str(Path(__file__).resolve())] = sha(Path(__file__))
    verify(pins)
    result = dict(
        status="VERIFIED_FIXED_MODEL_DIAGNOSIS_NOT_PRODUCT_COMPLETION",
        source_commit=first["source_commit"],
        receipt_hashes=dict(zip(NAMES, HASHES, strict=True)),
        paths={n: str(ROOT / n) for n in NAMES},
        input_pin_count=len(pins),
        provenance=first["provenance"],
        arms=first["arms"],
        training_constant_source_id=first["training_constant_source_id"],
        resources={
            n: {k: r[k] for k in ["seconds", "peak_cpu_bytes", "output_bytes"]}
            for n, r in zip(NAMES, reports, strict=True)
        },
        commands={n: r["command"] for n, r in zip(NAMES, reports, strict=True)},
        runtime={
            k: first[k]
            for k in [
                "python",
                "python_executable",
                "torch",
                "numpy",
                "device",
                "dtype",
                "cpu_threads",
                "interop_threads",
                "deterministic",
            ]
        },
        frozen_weights=True,
        model_updates=0,
        gradients_computed=False,
        gpu_used=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        scope="Observed nine-speaker development, fixed selected models; no blind result",
        recommendation=(
            "Close capacity/budget extension here; require genuinely joint decision task"
        ),
    )
    prefix = "R2_LOMBARD_VISUAL_STATE_RESULTS_20261015"
    (DOCS / (prefix + ".json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fixed-model visual state and mask diagnosis",
        "",
        "## Result and decision",
        "",
        "The primary and separate fresh process **actually completed** with byte-exact",
        "all24control prediction payloads, all training/validation latent arrays and",
        "constants, metrics/statistics/intervals. No learning, selection or calibration update.",
        "",
        "A single training-mean visual final state for every validation utterance preserves",
        "**all5334slot answers** in each joint model: broadcast99.53%, ordered99.51%.",
        "A fixed first training video or latent also preserves every answer. Matching visual",
        "content is unnecessary for these observed predictions. This supports a nearly",
        "constant learned-code explanation within this fixed-model development diagnosis,",
        "not a universal statement about multimodal models or a new audio-only product.",
        "",
        "In joint_ordered,100%of validation latent entries have abs(h)>=.99, average",
        "across-example channel std1.401e-5. Its visual contribution to the first readout",
        "affine layer has std2.639e-5 versus audio2.94249. Native visual inputs vary",
        "(std1.08201); the strongly attenuated variation is in the trained joint readout.",
        "The same frozen encoder's video-only model has latent std .61333 and affine",
        "contribution std1.11055; cached-token reversal reduces48.01%→21.18%.",
        "Do not diagnose an encoder failure from the joint branch collapse alone.",
        "",
        "Zeroing the latent lowers joint ordered99.51%→97.43%; clearing only its presence",
        "bit then gives97.34%. Thus nonzero learned visual state contributes to this",
        "parameterization, but matched visual semantics do not account for the answers.",
        "Equal registered parameter counts did not ensure equal active capacity in the",
        "earlier audio comparison. No capacity/epoch/auxiliary-loss sweep follows.",
        "",
        "Keep these scores and checkpoints in their scope. **No joint-Omni promotion**;",
        "next require a task where audio and candidate videos supply complementary",
        "evidence, with difficult source-matched distractors and unimodal controls.",
        "",
        "## All fixed controls (selected checkpoints, unchanged calibration)",
        "",
        "| Model | Control | Accuracy | NLL | Brier | ECE | Answer changes | "
        "Max raw logit change |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, r in first["arms"].items():
        for name, row in r["controls"].items():
            m = row["metrics"]["macro"]
            lines.append(
                f"| {arm} | {name} | {m['accuracy']:.2%} | {m['nll']:.9f} | "
                f"{m['brier']:.9f} | {m['ece']:.9f} | {row['slot_answer_changes']} | "
                f"{row['max_abs_raw_logit_change']:.6g} |"
            )
    lines += [
        "",
        "`input_zero` is the standardized pooled training mean repeated16times, not",
        "a natural blank video. `input_train_first` uses frozen training index0.",
        "Latent controls intervene at the existing readout boundary. Zero-present versus",
        "zero-absent holds audio/zero latent fixed and changes only the visual-presence bit.",
        "Mask/latent interventions are OOD; no natural missingness robustness claim.",
        "Reversal reorders16cached tokens, not raw-frame re-encoding; it diagnoses",
        "readout order dependence, not full physical temporal reasoning.",
        "",
        "## Integrity, uncertainty and preservation",
        "",
        "Models fixed in advance: joint_broadcast23/joint_ordered23/video_ordered18.",
        "All4080training and889development utterances, six full vocabularies/51options",
        "retained. Train means use FP64 accumulation→FP32, no targets. First training",
        f"source:`{first['training_constant_source_id']}`; this identity is recorded only,",
        "never a neural feature. Existing398calibration temperatures are unchanged.",
        "Native/decomposed readout logits are exact before any intervention.",
        "",
        "Paired whole-speaker10000bootstrap(seed17, linear percentile95%) of fixed",
        "train-mean-latent minus matched is in the adjacent JSON, along with all per-slot",
        "metrics. Accuracy/whole-command differences are exactly0on these samples.",
        "Nine observed development speakers, selected weights, single seed and artificial",
        "controls limit interpretation. No population-equivalence claim. NLL changes near",
        "1e-8–1e-6 are not presented as practically useful quality gains.",
        "",
        "106355registered parameters per sequentially loaded head, unchanged native",
        "encoder122573568/normalization2304 composition; encoders never loaded here.",
        "CPU i7-12700H,4threads/interOp1, FP32/deterministic, Python3.11.9,",
        "Torch2.6.0+cu124/NumPy2.4.6. Requires_grad flags retained for parity;",
        "eval/no_grad, no optimizer, gradients or updates. CPU tests overlapped the",
        "primary inference, so these are observed shared-PC wall times.",
        "",
        "| Stage | Seconds | Peak CPU GiB | New output MB | Receipt SHA |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, h, r in zip(NAMES, HASHES, reports, strict=True):
        lines.append(
            f"| {name} | {r['seconds']:.3f} | {r['peak_cpu_bytes'] / 2**30:.3f} | "
            f"{r['output_bytes'] / 1e6:.3f} | `{h}` |"
        )
    lines += [
        "",
        f"Source commit:`{first['source_commit']}`. Exact commands, runtime, checkpoint/",
        "corpus/cache hashes, state statistics and controls in",
        "[small results JSON](R2_LOMBARD_VISUAL_STATE_RESULTS_20261015.json).",
        "Both new output directories remain under`C:/CodexArtifacts/pretrained-reuse-r2/`.",
        "All source/runtime/checkpoint/receipt/control inputs are hash-bound and rechecked",
        "before/after execution, replay and publication. Resource guards600s/3GiB/64MiB/",
        "1GiB reserve held. No cloud/GPU/old8GiBguard changes, Teacher/sealed changes",
        "or legacy artifact overwrite. Weights/media/latent arrays stay outside Git.",
        "",
        "The user-withdrawn standalone <=10M audio program remains withdrawn. This",
        "diagnosis informs the required native-audio component of the main <=200M Omni",
        "research; it does not substitute another product goal. Four-modality95–98%,",
        "genuine joint/OOD/condition reuse and deployment gates remain unmet.",
    ]
    (DOCS / (prefix + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    verify(pins)
    print("Published verified fixed-state diagnosis; no product promotion")


if __name__ == "__main__":
    main()
