import argparse
import hashlib
import json
import pathlib
import time

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoImageProcessor, AutoModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
parser.add_argument("--model-dir", type=pathlib.Path, required=True)
parser.add_argument("--max-episodes", type=int)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()
if args.batch_size < 1:
    parser.error("--batch-size must be positive")
if args.device == "cuda" and not torch.cuda.is_available():
    parser.error("CUDA was requested but PyTorch reports no available CUDA device")
root = args.artifact_dir
model_dir = args.model_dir
receipt = json.loads((root / "download-receipt.json").read_text())
proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
torch.set_num_threads(4)
device = torch.device(args.device)
model.to(device)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
all_queries = []
decode_seconds = 0.0
infer_seconds = 0.0
train_ids = set(receipt.get("train_episode_indices", receipt.get("train_ids", [])))
validation_ids = set(
    receipt.get("validation_episode_indices", receipt.get("validation_ids", []))
)
video_files = [file_rec for file_rec in receipt["files"] if file_rec["kind"] == "video"]
if args.max_episodes is not None:
    if args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    video_files = video_files[: args.max_episodes]
for file_rec in video_files:
    if file_rec["kind"] != "video":
        continue
    ep = file_rec["episode"]
    if ep not in train_ids and ep not in validation_ids:
        raise ValueError(f"episode {ep} is not declared in train or validation receipt IDs")
    parquet_path = root / f"parquet-episode-{ep:06d}.parquet"
    rows = pq.read_table(parquet_path).to_pylist()
    if len(rows) < 12:
        raise RuntimeError(f"episode {ep} too short")
    timestamps = np.array([r["timestamp"] for r in rows], dtype=np.float64)
    frame_indexes = np.array([r["frame_index"] for r in rows], dtype=np.int64)
    if np.max(np.abs(timestamps - frame_indexes / 30.0)) > 1e-4:
        raise RuntimeError(f"timestamp/frame mismatch episode {ep}")
    n = len(rows)
    lo = 7
    hi = n - 4
    ts = sorted(set(int(round(lo + (hi - lo) * q)) for q in (0.2, 0.4, 0.6, 0.8)))
    wanted = {t - k for t in ts for k in range(7, -1, -1)}
    video_start = time.perf_counter()
    frames = {}
    local_video_path = file_rec.get("local_path", file_rec["path"])
    with av.open(local_video_path) as container:
        stream = container.streams.video[0]
        if stream.average_rate is None or abs(float(stream.average_rate) - 30) > 0.1:
            raise RuntimeError(f"fps mismatch {ep}")
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames[index] = frame.to_image().convert("RGB")
    if not wanted <= frames.keys():
        raise RuntimeError(f"missing selected frames ep={ep}")
    decode_seconds += time.perf_counter() - video_start
    q_images = []
    for t in ts:
        for j in range(t - 7, t + 1):
            q_images.append(frames[j])
    infer_start = time.perf_counter()
    frame_outputs = []
    for start in range(0, len(q_images), args.batch_size):
        inputs = proc(images=q_images[start : start + args.batch_size], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            frame_outputs.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    out = np.concatenate(frame_outputs, axis=0)
    infer_seconds += time.perf_counter() - infer_start
    feats = out.reshape(4, 8, -1).mean(axis=1)
    states = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
    actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
    for qi, t in enumerate(ts):
        all_queries.append(
            {
                "episode": ep,
                "split": "train" if ep in train_ids else "validation",
                "window": qi,
                "t": t,
                "video": feats[qi].astype(np.float32),
                "video_last": out.reshape(4, 8, -1)[qi, -1].astype(np.float32),
                "state": states[t - 7 : t + 1].reshape(-1),
                "action": actions[t + 3],
            }
        )
    print(
        json.dumps(
            {
                "episode": ep,
                "queries": len(ts),
                "decode_elapsed": decode_seconds,
                "inference_elapsed": infer_seconds,
            }
        ),
        flush=True,
    )
np.savez_compressed(
    root / "features.npz",
    video=np.stack([x["video"] for x in all_queries]),
    video_last=np.stack([x["video_last"] for x in all_queries]),
    state=np.stack([x["state"] for x in all_queries]),
    action=np.stack([x["action"] for x in all_queries]),
    episode=np.array([x["episode"] for x in all_queries]),
    split=np.array([x["split"] for x in all_queries]),
    window=np.array([x["window"] for x in all_queries]),
    t=np.array([x["t"] for x in all_queries]),
)
meta = {
    "queries": len(all_queries),
    "train_queries": sum(x["split"] == "train" for x in all_queries),
    "validation_queries": sum(x["split"] == "validation" for x in all_queries),
    "frames_per_query": 8,
    "decode_seconds": decode_seconds,
    "dinov2_cpu_seconds": infer_seconds,
    "device": args.device,
    "batch_size": args.batch_size,
    "peak_cuda_bytes": (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    ),
    "dtype": "float32",
    "model_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "source_revision": receipt.get("dataset_revision") or receipt["revision"],
    "timestamp_frame_verified": True,
    "reserved_holdout_accessed": False,
    "feature_file_sha256": hashlib.sha256((root / "features.npz").read_bytes()).hexdigest(),
}
(root / "feature-extraction.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta), flush=True)
