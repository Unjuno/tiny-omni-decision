"""Retain historical publication, measure stored-logit delta, correct reference wording."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.replay_native_av_correspondence import sha, verify

PHRASE = (
    "All selected native CPU video frame features and raw logits match the original\n"
    "native CPU bank exactly."
)


def correct_reference(text, delta):
    if text.count(PHRASE) != 1 or not np.isfinite(delta) or delta < 0:
        raise ValueError("exact prior reference text and finite measured delta required")
    return text.replace(
        PHRASE,
        "Native CPU video features match the original native framewise bank exactly.\n"
        "Raw video logits exactly match same-head recomputation from those cached\n"
        f"features. Versus original stored temporal-run logits, max difference is{delta:.9g}.\n"
        "Probability deltas below use same-head recomputation from original cached\n"
        "audio/video features, not original stored temporal-run predictions.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    previous = args.publication / "result.json"
    pins = {str(previous): args.publication_sha256}
    verify(pins)
    receipt = json.loads(previous.read_text(encoding="utf-8"))
    if receipt["status"] != "PASS_HASH_BOUND_NATIVE_AV_PUBLICATION":
        raise ValueError("complete prior publication required")
    pins.update(receipt["validated_hashes"])
    verify(pins)
    verify(receipt["output_hashes"])
    documents = {Path(p).suffix: Path(p) for p in receipt["output_hashes"]}
    summary = json.loads(documents[".json"].read_text(encoding="utf-8"))
    primary = Path(
        next(
            p
            for p in summary["receipt_hashes"]
            if Path(p).parent.name == "native-av-correspondence-runtime-01"
        )
    ).parent
    reference = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/lombard-temporal-full-01/video_ordered/best-train.npz"
    )
    if str(reference.resolve()) not in pins:
        raise ValueError("stored reference is not hash-bound")
    native = primary / "native-states.npz"
    if str(native.resolve()) not in pins:
        raise ValueError("native state arrays not hash-bound")
    with (
        np.load(reference, allow_pickle=False) as old,
        np.load(native, allow_pickle=False) as actual,
    ):
        indices = {str(s): i for i, s in enumerate(old["source_ids"])}
        selected = [indices[s] for s in summary["primary"]["selected_source_ids"]]
        per_row = np.abs(actual["video_logits"] - old["logits"][selected]).max(1).tolist()
        delta = max(per_row)
    for suffix, path in documents.items():
        history = args.output / ("historical" + suffix)
        history.write_bytes(path.read_bytes())
        pins[str(history.resolve())] = receipt["output_hashes"][str(path)]
    text = documents[".md"].read_text(encoding="utf-8")
    corrected = correct_reference(text, delta)
    summary.update(
        video_logit_reference="same-head B1 recomputation from original cached native features",
        probability_reference="same-head B1 recomputation from original cached audio/video features",
        stored_temporal_video_logit_comparison=dict(
            max_abs_delta=delta,
            per_selected_source_delta=per_row,
            reference_sha256=pins[str(reference.resolve())],
        ),
        publication_reference_correction=True,
    )
    pins[str(Path(__file__).resolve())] = sha(__file__)
    verify(pins)
    documents[".json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    documents[".md"].write_text(corrected, encoding="utf-8")
    verify(pins)
    final = dict(
        status="PASS_HASH_BOUND_CORRECTED_NATIVE_AV_PUBLICATION",
        previous_publication_sha256=args.publication_sha256,
        validated_hashes=pins,
        input_pin_count=len(pins),
        stored_temporal_video_max_abs_logit_delta=delta,
        output_hashes={str(p.resolve()): sha(p) for p in documents.values()},
    )
    (args.output / "result.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(final["status"])


if __name__ == "__main__":
    main()
