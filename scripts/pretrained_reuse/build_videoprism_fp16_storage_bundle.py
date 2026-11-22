"""Build an isolated FP16-storage / FP32-compute VideoPrism image bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tiny_omni_decision.bundle_validation import verify_bundle

SOURCE_BUNDLE_ID = "path-a-image-videoprism-clevr4-runtime-v0"
SOURCE_MANIFEST_SHA256 = "958e1f8735c33237342d0533f7529a0514fc4f002ca812057b2eff28ac51012e"
SOURCE_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
OUTPUT_BUNDLE_ID = "path-a-image-videoprism-clevr4-runtime-fp16-storage-v1"
CHECKPOINT_NAME = "videoprism-base-f16r288.npz"
RUNTIME_RELATIVE = Path("runtime/tiny_omni_decision/videoprism_runtime.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_npz_to_float16(source: Path, destination: Path) -> tuple[int, int]:
    """Cast every finite float32 array while retaining names and shapes."""
    with np.load(source, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("VideoPrism checkpoint is empty")
        converted: dict[str, np.ndarray] = {}
        for name in archive.files:
            value = archive[name]
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"expected finite float32 checkpoint tensor: {name}")
            with np.errstate(over="ignore", invalid="ignore"):
                half = value.astype(np.float16)
            if not np.isfinite(half).all():
                raise ValueError(f"FP16 conversion overflowed checkpoint tensor: {name}")
            converted[name] = half
    parameter_count = sum(int(value.size) for value in converted.values())
    np.savez(destination, **converted)
    return parameter_count, len(converted)


def _update_manifest(bundle: Path, manifest: dict[str, Any]) -> None:
    manifest["bundle_id"] = OUTPUT_BUNDLE_ID
    manifest["source_bundle_created_utc"] = manifest["created_utc"]
    manifest["created_utc"] = datetime.now(UTC).isoformat()
    manifest["precision_variant"] = {
        "storage_dtype": "float16",
        "inference_compute_dtype": "float32",
        "conversion": "all checkpoint arrays cast to IEEE FP16 on export and back to FP32 at load",
        "quality_status": "not_yet_evaluated",
    }
    checkpoint = bundle / CHECKPOINT_NAME
    runtime = bundle / RUNTIME_RELATIVE
    files = {entry["path"]: entry for entry in manifest["files"]}
    for relative, path in ((CHECKPOINT_NAME, checkpoint), (RUNTIME_RELATIVE.as_posix(), runtime)):
        entry = files[relative]
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    manifest["encoder"]["checkpoint_sha256"] = sha256_file(checkpoint)
    manifest["encoder"]["checkpoint_bytes"] = checkpoint.stat().st_size
    manifest["bundle_bytes_excluding_manifest"] = sum(
        int(entry["bytes"]) for entry in manifest["files"]
    )
    manifest["package_target_met"] = (
        manifest["bundle_bytes_excluding_manifest"] <= int(manifest["package_target_bytes"])
    )
    (bundle / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_bundle(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output bundle: {output}")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("source and output bundle paths must not contain one another")
    source_manifest_path = source / "bundle-manifest.json"
    if sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source image bundle manifest hash mismatch")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != SOURCE_BUNDLE_ID:
        raise ValueError("unexpected source bundle identity")
    checkpoint = source / CHECKPOINT_NAME
    if sha256_file(checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("source VideoPrism checkpoint hash mismatch")
    verify_bundle(source, manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary)
        parameter_count, tensor_count = convert_npz_to_float16(
            temporary / CHECKPOINT_NAME,
            temporary / "videoprism-fp16-storage.npz",
        )
        converted_checkpoint = temporary / "videoprism-fp16-storage.npz"
        original_checkpoint = temporary / CHECKPOINT_NAME
        original_checkpoint.unlink()
        converted_checkpoint.replace(original_checkpoint)

        runtime_path = temporary / RUNTIME_RELATIVE
        runtime = runtime_path.read_text(encoding="utf-8")
        if runtime.count(SOURCE_CHECKPOINT_SHA256) != 1:
            raise ValueError("source runtime checkpoint pin differs from the reviewed value")
        runtime = runtime.replace(SOURCE_CHECKPOINT_SHA256, sha256_file(original_checkpoint))
        if "self.model.apply(variables, video, train=False)" not in runtime:
            raise ValueError("source runtime inference expression differs from reviewed version")
        old_expression = (
            "self.encode = jax.jit(lambda video: "
            "self.model.apply(variables, video, train=False))"
        )
        new_expression = (
            "variables = jax.tree_util.tree_map(\n"
            "            lambda value: jax.numpy.asarray(\n"
            "                value, dtype=jax.numpy.float32\n"
            "            ), variables\n"
            "        )\n"
            "        self.encode = jax.jit(lambda video: "
            "self.model.apply(variables, video, train=False))"
        )
        runtime = runtime.replace(old_expression, new_expression)
        if "variables = jax.tree_util.tree_map(" not in runtime:
            raise ValueError("failed to install FP32 inference cast")
        old_bundle_id = f'"{SOURCE_BUNDLE_ID}"'
        if runtime.count(old_bundle_id) != 1:
            raise ValueError("source runtime bundle identity pin differs from the reviewed version")
        runtime_path.write_text(
            runtime.replace(old_bundle_id, f'"{OUTPUT_BUNDLE_ID}"'), encoding="utf-8"
        )

        manifest["source_bundle_id"] = SOURCE_BUNDLE_ID
        manifest["encoder"]["storage_dtype"] = "float16"
        manifest["encoder"]["inference_compute_dtype"] = "float32"
        manifest["encoder"]["checkpoint_parameter_count"] = parameter_count
        manifest["encoder"]["checkpoint_tensor_count"] = tensor_count
        _update_manifest(temporary, manifest)
        verify_bundle(temporary, manifest)
        temporary.rename(output)
        return {
            "bundle_id": OUTPUT_BUNDLE_ID,
            "parameter_count": parameter_count,
            "tensor_count": tensor_count,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "output_checkpoint_sha256": sha256_file(output / CHECKPOINT_NAME),
            "output_manifest_sha256": sha256_file(output / "bundle-manifest.json"),
            "bundle_bytes_excluding_manifest": manifest["bundle_bytes_excluding_manifest"],
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.source_bundle, args.output_bundle), indent=2))


if __name__ == "__main__":
    main()
