"""Verify completed grounding runs and quantify paired scene uncertainty."""

import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write


def read(path):
    return json.loads(path.read_text())


def compare(first, second):
    scenes = sorted({r["scene_index"] for r in first})
    kinds = sorted({r["taxonomy"] for r in first})
    counts = np.zeros((len(scenes), len(kinds)))
    sums = np.zeros((len(scenes), len(kinds), 3))
    for a, b in zip(first, second, strict=True):
        if any(a[k] != b[k] for k in ("id", "scene_index", "taxonomy", "target_index")):
            raise ValueError("unpaired predictions")
        vals = []
        for r in (a, b):
            p = np.array(r["probabilities"])
            t = r["target_index"]
            if not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-5:
                raise ValueError("invalid probabilities")
            vals.append(
                np.array(
                    [
                        p.argmax() == t,
                        -np.log(max(p[t], 1e-12)),
                        np.sum((p - np.eye(len(p))[t]) ** 2),
                    ]
                )
            )
        i, j = scenes.index(a["scene_index"]), kinds.index(a["taxonomy"])
        counts[i, j] += 1
        sums[i, j] += vals[1] - vals[0]
    indices = np.random.default_rng(17).integers(len(scenes), size=(10000, len(scenes)))
    denom = counts[indices].sum(1)
    if (denom == 0).any():
        raise ValueError("missing bootstrap taxonomy")
    boot = (sums[indices].sum(1) / denom[..., None]).mean(1)
    point = (sums.sum(0) / counts.sum(0)[:, None]).mean(0)
    return {
        k: dict(
            difference=float(point[i]),
            percentile_95_interval=np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
        )
        for i, k in enumerate(("accuracy", "nll", "brier"))
    }


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/video-grounding-training-01")
    result = read(root / "result.json")
    config = read(root / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    histories = {}
    for mode, r in result["conditions"].items():
        history = read(root / mode / "history.json")
        assert len(history) == 25 and history[-1]["epoch"] == 24 and r["reload_exact"]
        assert (
            min(history, key=lambda h: h["validation"]["macro_taxonomy"]["nll"])["epoch"]
            == r["best_epoch"]
        )
        for name in ("best", "final"):
            assert sha(root / mode / f"{name}.pt") == r[f"{name}_sha256"]
        histories[mode] = history
    video = read(root / "video/matched-predictions.json")
    zero = read(root / "no_video/matched-predictions.json")
    shuffled = read(root / "video/shuffled-predictions.json")
    assert zero == read(root / "no_video/shuffled-predictions.json")
    out = Path("docs/pretrained_reuse/R2_VIDEO_GROUNDING_RESULTS_20261012.json")
    if out.exists():
        raise FileExistsError(out)
    write(
        out,
        dict(
            result=result,
            config=config,
            histories=histories,
            comparisons=dict(
                video_minus_no_video=compare(zero, video),
                matched_minus_shuffled=compare(shuffled, video),
            ),
            bootstrap=dict(seed=17, draws=10000, unit="scene", aggregation="equal taxonomy"),
            hashes={str(p): sha(p) for p in root.rglob("*") if p.is_file()},
        ),
    )
    for mode, r in result["conditions"].items():
        print(mode, r["best_epoch"], r["selected"]["validation"]["macro_taxonomy"])
    print(json.dumps(read(out)["comparisons"], indent=2))


if __name__ == "__main__":
    main()
