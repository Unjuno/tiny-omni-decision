"""Pinned metadata and meta-device parameter audit before a costly checkpoint download."""

import json
import math
import struct
import subprocess
import urllib.request
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration

from scripts.range_generalization import sha, write


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main():
    out = Path("C:/CodexArtifacts/pretrained-reuse-r1/flan-t5-small-budget-audit-02")
    if out.exists():
        raise FileExistsError(out)
    info = json.loads(fetch("https://huggingface.co/api/models/google/flan-t5-small?blobs=true"))
    revision = info["sha"]
    out.mkdir()
    for name in ["config.json", "README.md"]:
        (out / name).write_bytes(
            fetch(f"https://huggingface.co/google/flan-t5-small/resolve/{revision}/{name}")
        )
    config = T5Config.from_dict(json.loads((out / "config.json").read_text()))
    with torch.device("meta"):
        model = T5ForConditionalGeneration(config)
        model.tie_weights()
    meta_parameters = sum(p.numel() for p in model.parameters())
    url = f"https://huggingface.co/google/flan-t5-small/resolve/{revision}/model.safetensors"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
    with urllib.request.urlopen(request, timeout=30) as response:
        prefix = response.read(65536)
    length = struct.unpack("<Q", prefix[:8])[0]
    if length > len(prefix) - 8:
        raise ValueError("bounded header incomplete")
    header = json.loads(prefix[8 : 8 + length])
    write(out / "header.json", header)
    parameters = sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    assert "shared.weight" in header and "lm_head.weight" in header
    assert not any("embed_tokens.weight" in k for k in header)

    shared = (
        model.shared.weight
        is model.encoder.embed_tokens.weight
        is model.decoder.embed_tokens.weight
    )
    assert shared and model.lm_head.weight is model.shared.weight
    towers = {"videoprism": 114365184, "whisper_tiny": 37760640, "dinov2_small": 22056576}
    files = {
        f["rfilename"]: dict(size=f.get("size"), lfs=f.get("lfs"))
        for f in info["siblings"]
        if f["rfilename"] in ["model.safetensors", "pytorch_model.bin"]
    }
    result = dict(
        model_id=info["id"],
        revision=revision,
        license=info.get("cardData", {}).get("license"),
        parameters_from_config_meta=meta_parameters,
        parameters_from_checkpoint_header=parameters,
        config_compatibility="FAIL: published untied output becomes tied; use checkpoint count",
        meta_embeddings_tied=shared,
        checkpoint_output_separate=True,
        loaded_pretrained_weights=False,
        quality_evaluated=False,
        current_video_audio_plus_text=parameters + towers["videoprism"] + towers["whisper_tiny"],
        current_video_audio_remaining_text_budget=200000000
        - towers["videoprism"]
        - towers["whisper_tiny"],
        static_image_audio_plus_text=parameters + towers["dinov2_small"] + towers["whisper_tiny"],
        caveat="Static DINO arithmetic omits a proven temporal model and fusion/head parameters.",
        files=files,
        hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R1_FLAN_T5_BUDGET_20261012.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
