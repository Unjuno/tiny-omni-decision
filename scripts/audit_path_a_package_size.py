"""Audit a manifest-listed Path A package and estimate a weight-only INT8 size."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ENCODER_FILES = (
    "components/image/videoprism-base-f16r288.npz",
    "components/audio/models/audio-whisper-tiny/model.safetensors",
    "components/audio/models/text-minilm-en/model.safetensors",
)


def _canonical_numpy_dtype(dtype: Any) -> str:
    canonical = {
        ("b", 1): "BOOL",
        ("f", 2): "F16",
        ("f", 4): "F32",
        ("f", 8): "F64",
        ("i", 1): "I8",
        ("i", 2): "I16",
        ("i", 4): "I32",
        ("i", 8): "I64",
        ("u", 1): "U8",
        ("u", 2): "U16",
        ("u", 4): "U32",
        ("u", 8): "U64",
    }
    try:
        return canonical[(dtype.kind, dtype.itemsize)]
    except KeyError as error:
        raise ValueError(f"unsupported NumPy checkpoint dtype: {dtype}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _listed_file(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe package path: {relative_path}")
    path = (root / Path(*relative.parts)).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"package path escapes assembly: {relative_path}")
    return path


def _validate_manifest_files(root: Path, manifest: dict[str, Any]) -> None:
    expected = set()
    for entry in manifest["files"]:
        relative = str(entry["path"])
        if relative in expected:
            raise ValueError(f"duplicate package path: {relative}")
        expected.add(relative)
        path = _listed_file(root, relative)
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"package size mismatch: {relative}")
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"package hash mismatch: {relative}")


def _npz_tensor_records(path: Path) -> list[dict[str, Any]]:
    import numpy.lib.format as npy_format

    records = []
    header_readers = {
        (1, 0): npy_format.read_array_header_1_0,
        (2, 0): npy_format.read_array_header_2_0,
    }
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as stream:
                version = npy_format.read_magic(stream)
                try:
                    read_header = header_readers[version]
                except KeyError as error:
                    raise ValueError(f"unsupported NPY header version: {version}") from error
                shape, _, dtype = read_header(stream)
            elements = math.prod(shape)
            records.append(
                {
                    "name": info.filename,
                    "dtype": _canonical_numpy_dtype(dtype),
                    "elements": elements,
                    "element_bytes": dtype.itemsize,
                    "stored_member_bytes": info.file_size,
                    "compressed_member_bytes": info.compress_size,
                }
            )
    return records


def _safetensors_records(path: Path) -> list[dict[str, Any]]:
    from safetensors import safe_open

    dtype_bytes = {
        "BOOL": 1,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "U16": 2,
        "I32": 4,
        "U32": 4,
        "I64": 8,
        "U64": 8,
    }
    records = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            dtype = tensor_slice.get_dtype()
            if dtype not in dtype_bytes:
                raise ValueError(f"unsupported safetensors dtype: {dtype}")
            records.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "elements": math.prod(tensor_slice.get_shape()),
                    "element_bytes": dtype_bytes[dtype],
                }
            )
    return records


def _estimate_int8(
    encoder_records: list[dict[str, Any]],
    *,
    group_size: int,
    fixed_package_bytes: int,
    encoder_file_bytes: int,
) -> dict[str, int]:
    if group_size < 1:
        raise ValueError("group size must be positive")
    float32_records = [record for record in encoder_records if record["dtype"] == "F32"]
    other_records = [record for record in encoder_records if record["dtype"] != "F32"]
    encoder_elements = sum(int(record["elements"]) for record in float32_records)
    float32_payload_bytes = sum(
        int(record["elements"]) * int(record["element_bytes"]) for record in float32_records
    )
    other_encoder_payload_bytes = sum(
        int(record["elements"]) * int(record["element_bytes"]) for record in other_records
    )
    preserved_file_overhead_bytes = (
        encoder_file_bytes - float32_payload_bytes - other_encoder_payload_bytes
    )
    if preserved_file_overhead_bytes < 0:
        raise ValueError("encoder tensor payload sizes exceed the current weight files")
    groups = sum(
        math.ceil(int(record["elements"]) / group_size) for record in float32_records
    )
    int8_payload_bytes = encoder_elements
    scale_bytes = groups * 4
    zero_point_bytes = groups
    estimate = (
        fixed_package_bytes
        + int8_payload_bytes
        + scale_bytes
        + zero_point_bytes
        + other_encoder_payload_bytes
        + preserved_file_overhead_bytes
    )
    return {
        "group_size": group_size,
        "float32_encoder_elements": encoder_elements,
        "non_float32_encoder_elements": sum(int(record["elements"]) for record in other_records),
        "group_count": groups,
        "int8_payload_bytes": int8_payload_bytes,
        "float32_scale_bytes": scale_bytes,
        "int8_zero_point_bytes": zero_point_bytes,
        "other_encoder_payload_bytes_preserved": other_encoder_payload_bytes,
        "existing_encoder_file_overhead_preserved": preserved_file_overhead_bytes,
        "fixed_package_bytes": fixed_package_bytes,
        "estimated_package_bytes": estimate,
    }


def audit_package(assembly: Path, *, group_size: int = 256) -> dict[str, Any]:
    root = assembly.resolve(strict=True)
    manifest_path = root / "package-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    _validate_manifest_files(root, manifest)

    file_records = {str(row["path"]): row for row in manifest["files"]}
    missing = sorted(set(ENCODER_FILES) - set(file_records))
    if missing:
        raise ValueError(f"encoder files missing from package manifest: {missing}")

    encoder_records = []
    encoder_files = []
    for relative in ENCODER_FILES:
        path = _listed_file(root, relative)
        if relative.endswith(".npz"):
            tensors = _npz_tensor_records(path)
        elif relative.endswith(".safetensors"):
            tensors = _safetensors_records(path)
        else:
            raise ValueError(f"unsupported encoder file: {relative}")
        encoder_records.extend(tensors)
        encoder_files.append(
            {
                "path": relative,
                "bytes": file_records[relative]["bytes"],
                "sha256": file_records[relative]["sha256"],
                "tensor_count": len(tensors),
                "dtype_counts": {
                    dtype: sum(row["dtype"] == dtype for row in tensors)
                    for dtype in sorted({row["dtype"] for row in tensors})
                },
                "elements_by_dtype": {
                    dtype: sum(
                        int(row["elements"]) for row in tensors if row["dtype"] == dtype
                    )
                    for dtype in sorted({row["dtype"] for row in tensors})
                },
            }
        )

    package_bytes = sum(int(row["bytes"]) for row in manifest["files"])
    encoder_file_bytes = sum(int(file_records[path]["bytes"]) for path in ENCODER_FILES)
    fixed_package_bytes = package_bytes - encoder_file_bytes
    parameters = int(manifest["unique_runtime_parameters"])
    estimate = _estimate_int8(
        encoder_records,
        group_size=group_size,
        fixed_package_bytes=fixed_package_bytes,
        encoder_file_bytes=encoder_file_bytes,
    )
    estimated_parameters = parameters

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "assembly_path": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_file_count": len(manifest["files"]),
        "package_content_bytes_excluding_root_manifest": package_bytes,
        "manifest_parameter_count": parameters,
        "encoder_files": encoder_files,
        "encoder_file_bytes": encoder_file_bytes,
        "fixed_package_bytes_outside_encoder_weights": fixed_package_bytes,
        "storage_only_int8_estimate": estimate,
        "estimated_loaded_parameter_count": estimated_parameters,
        "assumptions": [
            (
                "Every FP32 tensor in the three listed encoder checkpoint files is stored as "
                "one INT8 value per element."
            ),
            (
                "One FP32 scale and one INT8 zero point are stored per independent "
                "256-element tensor group."
            ),
            (
                "Non-FP32 tensors and every package file outside those three encoder "
                "checkpoint files keep their current bytes."
            ),
            (
                "This estimate excludes container/filesystem padding and any additional "
                "runtime-only metadata."
            ),
            (
                "No quantized files, kernels, model outputs, or quality metrics were "
                "produced or tested."
            ),
        ],
        "sealed_audit_loaded": False,
        "training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=256)
    args = parser.parse_args()
    output = args.output_report.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite size audit output: {output}")
    report = audit_package(args.assembly, group_size=args.group_size)
    repository = Path(__file__).resolve().parents[1]
    report["source_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report["source_branch"] = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report["runner_sha256"] = _sha256(Path(__file__).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    digest = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(
        json.dumps(
            {"report": str(output), "sha256": digest, **report["storage_only_int8_estimate"]}
        )
    )


if __name__ == "__main__":
    main()
