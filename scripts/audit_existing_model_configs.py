"""Read-only local config and real checkpoint load reconciliation."""

import gc
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    WhisperForConditionalGeneration,
)

from scripts.range_generalization import sha, write

FIELDS = [
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "tie_word_embeddings",
    "hidden_act",
    "attention_bias",
    "mlp_bias",
    "rms_norm_eps",
    "layer_norm_eps",
    "layer_norm_epsilon",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "rope_interleaved",
    "patch_size",
    "image_size",
    "num_channels",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "layer_norm_eps",
    "d_model",
    "encoder_layers",
    "decoder_layers",
    "encoder_attention_heads",
    "decoder_attention_heads",
    "encoder_ffn_dim",
    "decoder_ffn_dim",
    "num_mel_bins",
    "max_source_positions",
    "max_target_positions",
    "scale_embedding",
    "activation_function",
]


def main():
    r1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")
    r2 = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    out = r1 / "existing-config-reconciliation-01"
    if out.exists():
        raise FileExistsError(out)
    specs = [
        ("dinov2", r1 / "dinov2-small-audit-01/model", AutoModel, 22056576),
        (
            "minilm",
            r2 / "path-a-audio-multiquery-runtime-bundle-v2/models/text-minilm-en",
            AutoModel,
            17389824,
        ),
        (
            "crossencoder",
            r1 / "crossencoder-minilm-l6-audit-01/model",
            AutoModelForSequenceClassification,
            22713601,
        ),
        ("smollm2", r1 / "smollm2-135m-audit-01/model", AutoModelForCausalLM, 134515008),
        (
            "whisper",
            r2 / "path-a-audio-multiquery-runtime-bundle-v2/models/audio-whisper-tiny",
            WhisperForConditionalGeneration,
            37760640,
        ),
    ]
    torch.set_num_threads(2)
    out.mkdir()
    results = {}
    for name, path, loader, expected in specs:
        start = time.perf_counter()
        raw = json.loads((path / "config.json").read_text())
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        differences = {
            k: dict(published=raw[k], effective=getattr(config, k, None))
            for k in FIELDS
            if k in raw and raw[k] != getattr(config, k, None)
        }
        model, info = loader.from_pretrained(
            path, local_files_only=True, dtype=torch.float32, output_loading_info=True
        )
        model.requires_grad_(False).eval()
        actual = sum(p.numel() for p in model.parameters())
        problems = {k: v for k, v in info.items() if v}
        loaded_differences = {
            k: dict(published=raw[k], effective=getattr(model.config, k, None))
            for k in FIELDS
            if k in raw and raw[k] != getattr(model.config, k, None)
        }
        results[name] = dict(
            path=str(path),
            parameters=actual,
            expected_parameters=expected,
            count_matches=actual == expected,
            config_differences=differences,
            loaded_config_differences=loaded_differences,
            load_diagnostics=problems,
            dtype=str(next(model.parameters()).dtype),
            seconds=time.perf_counter() - start,
            config_sha256=sha(path / "config.json"),
            weights_sha256=sha(path / "model.safetensors"),
        )
        print(name, actual, differences, problems, flush=True)
        write(out / "progress.json", results)
        del model
        gc.collect()
    result = dict(
        models=results,
        torch=torch.__version__,
        transformers=transformers.__version__,
        device="cpu",
        training=False,
        inference=False,
        all_counts_match=all(r["count_matches"] for r in results.values()),
    )
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R1_EXISTING_CONFIG_RECONCILIATION_20261012.json"), result)


if __name__ == "__main__":
    main()
