"""Raw image features to folded readout and deterministic typed JSON replay."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_dinov2_image_probe import sha, write
from tiny_omni_decision.bilinear_readout import BilinearCandidateScorer
from tiny_omni_decision.classification_json import classification_json


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    raw = root / "shared-global-raw-image-replay-01"
    folded = root / "shared-global-folded-readout-01"
    parent = root / "shared-spatial-image-readout-01"
    result = json.loads((raw / "result.json").read_text())
    assert result["status"] == "PASS_RAW_FEATURES"
    assert sha(raw / "cells.npy") == result["features_sha256"]
    fold_report = json.loads((folded / "result.json").read_text())
    assert sha(folded / "readout.pt") == fold_report["weights_sha256"]
    assert sha(parent / "option-embeddings.npy") == json.loads(
        (parent / "config.json").read_text())["option_embeddings_sha256"]
    out = root / "shared-global-json-replay-01"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    model = BilinearCandidateScorer(1536, 384, 128).eval()
    model.load_state_dict(torch.load(folded / "readout.pt", weights_only=True, map_location="cpu"))
    cells = np.load(raw / "cells.npy", allow_pickle=False)
    feature = torch.from_numpy(np.concatenate((cells[:, 0], cells[:, 1:].mean(1)), 1))
    options = torch.from_numpy(np.load(parent / "option-embeddings.npy", allow_pickle=False))
    ids = torch.arange(32)
    inputs = torch.cat((feature[ids // 4, None].expand(-1, 10, -1), options[ids % 4]), -1)
    with torch.inference_mode():
        probabilities = model(inputs).softmax(-1).numpy()
    saved = json.loads((parent / "global/matched-predictions.json").read_text())[:32]
    raw_records = json.loads((raw / "config.json").read_text())["rows"]
    assert [r["image_id"] for r in saved] == [r["id"] for r in raw_records for _ in range(4)]
    np.testing.assert_allclose(probabilities, [r["probabilities"] for r in saved],
                               atol=1e-5, rtol=0)
    outputs = []
    for row, probs in zip(saved, probabilities, strict=True):
        supplied = [dict(option_id=f"custom-{i}", class_id=name)
                    for i, name in enumerate(row["options"])]
        decision = classification_json(probs, row["options"], supplied)
        reversed_decision = classification_json(probs, row["options"], supplied[::-1])
        assert decision["selected_option_id"] == reversed_decision["selected_option_id"]
        assert decision["selected_option_id"] == supplied[
            row["options"].index(row["prediction"])]["option_id"]
        outputs.append(dict(id=row["id"], decision=decision))
    write(out / "decisions.json", outputs)
    write(out / "result.json", dict(status="PASS_RAW_TO_JSON", images=8, decisions=32,
        changed_answers=0, options_reordered_without_answer_changes=True,
        max_probability_delta=float(np.abs(probabilities-np.array(
            [r["probabilities"] for r in saved])).max()),
        script_sha256=sha(Path(__file__)), decisions_sha256=sha(out / "decisions.json"),
        raw_features_sha256=result["features_sha256"], weights_sha256=fold_report["weights_sha256"],
        scope="fixed 10-class taxonomies only; not arbitrary semantic options or joint fusion"))
    print("PASS_RAW_TO_JSON: 8 images, 32 decisions, unchanged answers")


if __name__ == "__main__":
    main()
