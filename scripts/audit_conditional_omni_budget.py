"""Pinned file-level budget for the current semantic/native candidate components."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import zipfile
from pathlib import Path

import numpy as np

CONFIG = "configs/pretrained_reuse/conditional_omni_budget_v1.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_record(shape, dtype, itemsize):
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid tensor shape")
    count = math.prod(shape)
    return dict(shape=list(shape), dtype=str(dtype), elements=count, tensor_bytes=count * itemsize)


def array_inventory(path):
    path = Path(path)
    tensors = {}
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("duplicate tensor entries")
            for name in archive.namelist():
                if not name.endswith(".npy"):
                    raise ValueError("only numeric tensor entries permitted")
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError("unsupported numpy header version")
                    shape, _, dtype = reader(stream)
                    if dtype.hasobject or dtype.kind not in "biufc":
                        raise ValueError("object/non-numeric tensor prohibited")
                    tensors[name] = tensor_record(shape, dtype, dtype.itemsize)
    elif path.suffix == ".safetensors":
        itemsize = dict(F64=8, F32=4, F16=2, BF16=2, I64=8, I32=4, I16=2, I8=1,
                        U64=8, U32=4, U16=2, U8=1, BOOL=1)
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 16 * 2**20:
                raise ValueError("unreasonable safetensors header")
            header = json.loads(stream.read(length))
        for name, value in header.items():
            if name == "__metadata__":
                continue
            record = tensor_record(value["shape"], value["dtype"], itemsize[value["dtype"]])
            start, end = value["data_offsets"]
            if (start < 0 or end - start != record["tensor_bytes"]
                    or end + 8 + length > path.stat().st_size):
                raise ValueError("invalid tensor offsets")
            tensors[name] = record
    elif path.suffix == ".pt":
        import torch

        values = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(values, dict) or not all(isinstance(v, torch.Tensor)
                                                  for v in values.values()):
            raise ValueError("flat tensor-only checkpoint required")
        tensors = {name: tensor_record(tuple(value.shape), value.dtype, value.element_size())
                   for name, value in values.items()}
    else:
        raise ValueError("unsupported tensor inventory format")
    if not tensors:
        raise ValueError("empty tensor inventory")
    return dict(tensors=tensors, tensor_count=len(tensors),
                elements=sum(v["elements"] for v in tensors.values()),
                tensor_bytes=sum(v["tensor_bytes"] for v in tensors.values()))


def summarize(components, calibration_count):
    count = sum(row["elements"] for row in components.values()) + calibration_count
    size = sum(row["tensor_bytes"] for row in components.values()) + 8 * calibration_count
    return dict(stored_tensor_elements_plus_calibration=count,
                tensor_bytes_plus_float64_calibration=size,
                headroom_elements=200_000_000 - count, within_200m_upper_bound=count <= 200_000_000,
                combined_model_loaded=False, quality_qualified=False,
                package_bytes_measured=False, native_forwards=0, optimizer_updates=0,
                count_scope="all declared tensor elements including buffers and calibration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
    pins = {entry["path"]: entry["sha256"] for entry in config["components"].values()}
    calibration = config["calibration_metadata"]
    pins[calibration["path"]] = calibration["sha256"]
    pins[str(Path(CONFIG).resolve())] = sha(CONFIG)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("input/source hash mismatch")
    components = {name: dict(path=value["path"], sha256=value["sha256"],
                              file_bytes=Path(value["path"]).stat().st_size,
                              **array_inventory(value["path"]))
                  for name, value in config["components"].items()}
    metadata = json.loads(Path(calibration["path"]).read_text(encoding="utf-8"))
    if metadata["status"] != "PASS_FROZEN_NATIVE_CORRESPONDENCE":
        raise ValueError("unverified calibration metadata")
    temperatures = {arm: metadata["models"][arm]["temperatures"]
                    for arm in ("audio", "video_ordered")}
    temperatures["query"] = [metadata["query_temperature"]]
    if any(not math.isfinite(v) or v <= 0 for values in temperatures.values() for v in values):
        raise ValueError("invalid calibration coefficients")
    summary = summarize(components, sum(len(v) for v in temperatures.values()))
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("source changed during inventory")
    result = dict(status="PASS_COMPONENT_FILE_INVENTORY", config=config, input_pins=pins,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  components=components, calibration=temperatures, summary=summary,
                  gpu_used=False, sealed_audit_loaded=False, transfer_loaded=False,
                  whole_omni_quality_unverified=True,
                  remaining_gates=["natural condition transfer", "native per-task quality",
                                   "general temporal reasoning", "complete integration",
                                   "Japanese task quality", "runtime and quantized package"],
                  adaptation_delta_is_replacement_not_added_parameters=True)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    print(json.dumps(dict(output=str(args.output), result_sha256=sha(path), summary=summary)))


if __name__ == "__main__":
    main()
