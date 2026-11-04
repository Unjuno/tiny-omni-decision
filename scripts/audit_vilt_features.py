"""Answer coverage and two-training-image conditioned-feature audit; no quality metric."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

from scripts.audit_vilt_candidate import WEIGHTS_SHA
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.vilt_compat import remove_legacy_position_ids

QUESTIONS = {
    "color": "What color are the objects?",
    "texture": "What texture do the objects have?",
    "count": "How many objects are there?",
    "shape": "What shape are the objects?",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r1/vilt-feature-audit-01")
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/vilt-vqa-audit-01/model")
    data = Path("C:/CodexArtifacts/pretrained-reuse-r2/clevr4-full-coverage-seed17")
    manifest = data / "train-image-manifest.jsonl"
    if root.exists():
        raise FileExistsError(root)
    if sha(model_path / "pytorch_model.bin") != WEIGHTS_SHA or sha(manifest) != (
        "2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010"
    ):
        raise ValueError("pinned inputs changed")
    config = ViltConfig.from_pretrained(model_path, local_files_only=True)
    coverage = {
        kind: {
            "supported": [str(x) for x in choices if str(x) in config.label2id],
            "unsupported": [str(x) for x in choices if str(x) not in config.label2id],
        }
        for kind, choices in CLEVR4_TAXONOMIES.items()
    }
    rows = [json.loads(line) for line in manifest.read_text().splitlines()][:2]
    root.mkdir()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    model = ViltForQuestionAnswering(config)
    state = torch.load(model_path / "pytorch_model.bin", weights_only=True, map_location="cpu")
    model.load_state_dict(
        remove_legacy_position_ids(state, config.max_position_embeddings), strict=True
    )
    del state
    model.eval().requires_grad_(False).cuda()
    processor = ViltProcessor.from_pretrained(model_path, local_files_only=True)
    lengths = [len(x) for x in processor.tokenizer(list(QUESTIONS.values()))["input_ids"]]
    if max(lengths) > config.max_position_embeddings:
        raise ValueError("questions exceed position table")
    records, arrays = [], []
    for row in rows:
        path = data / "train" / "images" / (row["id"] + ".png")
        if sha(path) != row["sha256"]:
            raise ValueError("media changed")
        with Image.open(path) as source:
            image = source.convert("RGB")
        inputs = processor(
            images=[image] * 4,
            text=list(QUESTIONS.values()),
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to("cuda")
        tick = time.perf_counter()
        with torch.inference_mode():
            features = model.vilt(**inputs).pooler_output
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        values = features.cpu().numpy()
        if values.shape != (4, 768) or not np.isfinite(values).all():
            raise ValueError("invalid frozen features")
        arrays.append(values)
        records.append(
            dict(
                id=row["id"],
                image_sha256=row["sha256"],
                input_shapes={k: list(v.shape) for k, v in inputs.items()},
                forward_seconds=elapsed,
                feature_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            )
        )
    values = np.stack(arrays)
    np.save(root / "features.npy", values)
    np.testing.assert_array_equal(np.load(root / "features.npy", allow_pickle=False), values)
    result = dict(
        status="PASS_FEATURE_PATH_ONLY",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        checkpoint_sha256=WEIGHTS_SHA,
        questions=QUESTIONS,
        question_token_lengths=lengths,
        answer_coverage=coverage,
        records=records,
        feature_shape=list(values.shape),
        file_sha256=sha(root / "features.npy"),
        total_loaded_parameters=sum(p.numel() for p in model.parameters()),
        feature_encoder_parameters=sum(p.numel() for p in model.vilt.parameters()),
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        projected_8424_train_512_val_4questions_array_bytes=(8424 + 512) * 4 * 768 * 4,
        estimated_full_forward_seconds=records[-1]["forward_seconds"] * (8424 + 512),
        estimation_limitation="One warm batch; excludes decode/I/O and throughput variation.",
        training=False,
        quality_evaluation=False,
        cost_usd=0,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
