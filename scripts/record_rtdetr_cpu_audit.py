"""Recheck real CPU/fresh artifacts and publish compact mechanics-only evidence."""

import json
from pathlib import Path

import numpy as np

from scripts.audit_rtdetr_cpu_candidate import MODEL_ID, REVISION, ROOT, WEIGHT_SHA, sha


def main():
    runs = []
    protected = {}
    banks = []
    for name in ["runtime-01", "runtime-reload-01"]:
        folder = ROOT / name
        path = folder / "result.json"
        r = json.loads(path.read_bytes())
        if (
            r["status"] != "PASS_R1_CPU_MECHANICS_ONLY"
            or r["model_id"] != MODEL_ID
            or r["revision"] != REVISION
            or r["weight_sha256"] != WEIGHT_SHA
            or r["cuda_initialized"]
            or r["training"]
            or r["annotations_loaded"]
            or not r["parameters_frozen"]
            or not r["cpu_rng_unchanged"]
            or r["state_before"] != r["state_after"]
            or r["queries_retained"] != 300
            or r["classes_retained"] != 80
            or r["parameters"] + r["registered_buffer_elements"]
            != r["checkpoint"]["tensor_elements"]
        ):
            raise ValueError("complete pinned CPU mechanics/parameter/source scope required")
        source = folder / "execution-source.py"
        if sha(source) != r["execution_source_sha256"]:
            raise ValueError("actual execution snapshot changed")
        output = folder / "all-query-outputs.npz"
        if sha(output) != r["outputs_sha256"]:
            raise ValueError("actual all-query artifact changed")
        with np.load(output, allow_pickle=False) as b:
            a = {k: b[k] for k in b.files}
        expected = {
            "input_pixel_values": (1, 3, 640, 640),
            "logits": (1, 300, 80),
            "boxes": (1, 300, 4),
        }
        if set(a) != set(expected) or any(
            v.shape != expected[k] or v.dtype != np.float32 or not np.isfinite(v).all()
            for k, v in a.items()
        ):
            raise ValueError("all native input/query/class arrays required")
        if (a["boxes"] < 0).any() or (a["boxes"] > 1).any():
            raise ValueError("native normalized boxes outside bounds")
        banks.append(a)
        protected.update(r["protected_hashes"])
        protected.update(
            {
                str(path): sha(path),
                str(source): r["execution_source_sha256"],
                str(output): r["outputs_sha256"],
            }
        )
        compact = {
            k: v for k, v in r.items() if k not in {"checkpoint", "protected_hashes", "processor"}
        }
        runs.append(dict(path=str(path), sha256=sha(path), summary=compact))
    for k in banks[0]:
        if banks[0][k].tobytes() != banks[1][k].tobytes():
            raise ValueError("fresh model native outputs differ from original")
    a, b = [x["summary"] for x in runs]
    for key in [
        "state_before",
        "input",
        "execution_source_sha256",
        "parameters",
        "parameter_bytes",
    ]:
        if a[key] != b[key]:
            raise ValueError("fresh input/model/provenance differs")
    for p, h in protected.items():
        if sha(p) != h:
            raise ValueError("protected provider/source/library/snapshot changed")
    report = dict(
        status="PASS_R1_CPU_FRESH_MECHANICS_ONLY",
        model_id=MODEL_ID,
        revision=REVISION,
        weight_sha256=WEIGHT_SHA,
        runs=runs,
        unique_registered_parameters=a["parameters"],
        registered_buffer_elements=a["registered_buffer_elements"],
        checkpoint_tensor_elements=a["parameters"] + a["registered_buffer_elements"],
        released_model_files_bytes=sum(
            (ROOT / "model" / n).stat().st_size
            for n in ["README.md", "config.json", "preprocessor_config.json", "model.safetensors"]
        ),
        released_weight_license="apache-2.0",
        model_policy="ALLOW",
        primary_model_card=f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/README.md",
        fresh_model_all_query_input_shape_dtype_bytes_equal=True,
        source_code_uncommitted_at_execution=True,
        publication_pending_active_native_job=True,
        no_GPU_used=True,
        training=False,
        decision_quality=False,
        automatic_semantic_discovery_quality=False,
        product_adopted=False,
        full_Omni_quality=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        protected_hashes=protected,
        recorder_sha256=sha(__file__),
    )
    target = Path("docs/pretrained_reuse/R1_RTDETR_V2_R18_CPU_RESULTS_20261014.json")
    if target.exists():
        raise ValueError("new publication snapshot required")
    target.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    print(
        report["status"],
        "parameters",
        report["unique_registered_parameters"],
        "files",
        report["released_model_files_bytes"],
        flush=True,
    )


if __name__ == "__main__":
    main()
