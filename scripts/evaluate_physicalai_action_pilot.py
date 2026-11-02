import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
args = parser.parse_args()
root = args.artifact_dir
d = np.load(root / "features.npz", allow_pickle=False)
tr = d["split"] == "train"
va = d["split"] == "validation"
torch.manual_seed(17)
np.random.seed(17)
torch.set_num_threads(2)
actions = d["action"].astype(np.float64)
mu = actions[tr].mean(0)
sd = actions[tr].std(0)
sd = np.maximum(sd, 1e-5)
z = (actions - mu) / sd
train_z = z[tr]
# deterministic farthest-first medoids/prototypes, fit only on train targets
centers = [int(np.argmax(np.sum((train_z - train_z.mean(0)) ** 2, axis=1)))]
while len(centers) < 8:
    dist = np.min(((train_z[:, None, :] - train_z[centers][None, :, :]) ** 2).sum(2), axis=1)
    centers.append(int(np.argmax(dist)))
C = train_z[centers].copy()
for _ in range(100):
    labels = np.argmin(((train_z[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
    new = C.copy()
    for k in range(8):
        if np.any(labels == k):
            new[k] = train_z[labels == k].mean(0)
    if np.allclose(C, new):
        break
    C = new
y = np.argmin(((z[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)


def metrics(prob, ytrue):
    pred = prob.argmax(1)
    acc = float((pred == ytrue).mean())
    nll = float(-np.log(np.clip(prob[np.arange(len(ytrue)), ytrue], 1e-8, 1)).mean())
    one = np.eye(prob.shape[1])[ytrue]
    brier = float(((prob - one) ** 2).sum(1).mean())
    conf = prob.max(1)
    bins = []
    for a in np.linspace(0, 1, 11)[:-1]:
        mask = (conf >= a) & (conf < min(a + 0.1, 1.000001))
        if mask.any():
            bins.append(
                (
                    float(mask.mean()),
                    float(abs((pred[mask] == ytrue[mask]).mean() - conf[mask].mean())),
                )
            )
    ece = float(sum(w * e for w, e in bins))
    return {"accuracy": acc, "nll": nll, "brier": brier, "ece": ece, "n": len(ytrue)}


def fit_eval(name, Xtr, Xva, ytr, yva):
    m = Xtr.mean(0)
    s = Xtr.std(0).clamp_min(1e-5)
    Xtr = (Xtr - m) / s
    Xva = (Xva - m) / s
    model = nn.Linear(Xtr.shape[1], 8)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = model(Xva).softmax(1).numpy()
    return metrics(p, yva.numpy()), p, (model, m, s)


yt = torch.tensor(y[tr], dtype=torch.long)
yv = torch.tensor(y[va], dtype=torch.long)
views = {
    "proprio_history": ("state", "state"),
    "video_8frame_mean": ("video", "video"),
    "video_last_frame": ("video_last", "video_last"),
    "joint_video_proprio": ("joint", "joint"),
}
results = {}
predictions = {}
fitted = {}
for name, (a, _b) in views.items():
    X = np.concatenate([d["video"], d["state"]], axis=1) if a == "joint" else d[a]
    res, p, fit = fit_eval(
        name,
        torch.tensor(X[tr], dtype=torch.float32),
        torch.tensor(X[va], dtype=torch.float32),
        yt,
        yv,
    )
    results[name] = res
    predictions[name] = p
    fitted[name] = fit
# Pair each held-out episode's proprio history with the next held-out episode's video,
# preserving within-episode window order while breaking video/state identity.
val_eps = d["episode"][va]
val_windows = d["window"][va]
unique_eps = np.unique(val_eps)
next_ep = {e: unique_eps[(i + 1) % len(unique_eps)] for i, e in enumerate(unique_eps)}
lookup = {
    (int(e), int(w)): d["video"][va][idx]
    for idx, (e, w) in enumerate(zip(val_eps, val_windows, strict=True))
}
misaligned_video = np.stack(
    [lookup[(int(next_ep[e]), int(w))] for e, w in zip(val_eps, val_windows, strict=True)]
)
model, mean, std = fitted["joint_video_proprio"]
Xmis = np.concatenate([misaligned_video, d["state"][va]], axis=1)
with torch.no_grad():
    p_misaligned = model((torch.tensor(Xmis, dtype=torch.float32) - mean) / std).softmax(1).numpy()
results["joint_video_episode_misaligned"] = metrics(p_misaligned, y[va])
predictions["joint_video_episode_misaligned"] = p_misaligned
# option-only class prior
prior = np.bincount(y[tr], minlength=8).astype(float)
prior /= prior.sum()
results["train_prior_option_only"] = metrics(np.tile(prior, (int(va.sum()), 1)), y[va])
results["train_class_counts"] = np.bincount(y[tr], minlength=8).tolist()
results["validation_class_counts"] = np.bincount(y[va], minlength=8).tolist()
# Codebook error against selected future actions
nearest = C[y]
residual = z - nearest
results["action_codebook"] = {
    "train_normalized_rmse": float(np.sqrt(np.mean(residual[tr] ** 2))),
    "validation_normalized_rmse": float(np.sqrt(np.mean(residual[va] ** 2))),
    "prototype_count": 8,
    "prototype_fit": "train-only deterministic farthest-first + Lloyd means",
    "dim_scales_train_only": sd.tolist(),
}
# clustered paired bootstrap for joint vs single-modality accuracy differences
rng = np.random.default_rng(17)
episodes = d["episode"][va]
unique = np.unique(episodes)
yv_np = y[va]
pred_joint = predictions["joint_video_proprio"].argmax(1)
boot = {}
for name in (
    "proprio_history",
    "video_8frame_mean",
    "video_last_frame",
    "joint_video_episode_misaligned",
):
    pred_base = predictions[name].argmax(1)
    diffs = []
    for _ in range(2000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(episodes == e) for e in chosen])
        diffs.append(
            float((pred_joint[ix] == yv_np[ix]).mean() - (pred_base[ix] == yv_np[ix]).mean())
        )
    boot[name] = {
        "joint_minus_baseline_accuracy_mean": float(np.mean(diffs)),
        "episode_cluster_bootstrap_95pct": np.quantile(diffs, [0.025, 0.975]).tolist(),
        "resamples": 2000,
        "seed": 17,
    }
results["episode_cluster_bootstrap_joint_accuracy"] = boot
results["protocol"] = {
    "train_episodes": 64,
    "validation_episodes": 16,
    "windows_per_episode": 4,
    "train_queries": int(tr.sum()),
    "validation_queries": int(va.sum()),
    "future_action_offset_frames": 3,
    "fps": 30,
    "target": "quantized demonstrator-action prediction; not optimality/success",
    "sealed_or_reserved_holdout_used": False,
    "validation_used_for_checkpoint_selection": False,
    "seed": 17,
    "classifier": "linear softmax; train-only z-score; LBFGS max_iter=200",
}
validation_predictions = []
for row_index, feature_index in enumerate(np.flatnonzero(va)):
    prediction = {
        "episode_index": int(d["episode"][feature_index]),
        "window_index": int(d["window"][feature_index]),
        "frame_index": int(d["t"][feature_index]),
        "target_prototype": int(y[feature_index]),
    }
    for arm, probabilities in predictions.items():
        prediction[arm] = {
            "prediction": int(probabilities[row_index].argmax()),
            "probabilities": probabilities[row_index].tolist(),
        }
    prediction["target_action"] = d["action"][feature_index].tolist()
    validation_predictions.append(prediction)
predictions_path = root / "validation-predictions.json"
predictions_path.write_text(json.dumps(validation_predictions, indent=2) + "\n")
checkpoint = {
    "seed": 17,
    "action_mean_train": torch.tensor(mu, dtype=torch.float64),
    "action_std_train": torch.tensor(sd, dtype=torch.float64),
    "action_prototypes_train": torch.tensor(C, dtype=torch.float64),
    "arms": {
        name: {
            "state_dict": {key: value.detach().cpu() for key, value in fit[0].state_dict().items()},
            "feature_mean_train": fit[1],
            "feature_std_train": fit[2],
        }
        for name, fit in fitted.items()
    },
}
torch.save(checkpoint, root / "probe-checkpoints.pt")
reloaded_checkpoint = torch.load(
    root / "probe-checkpoints.pt", map_location="cpu", weights_only=True
)
reload_max_abs_diff = {}
for name, source_features in (
    ("proprio_history", d["state"][va]),
    ("video_8frame_mean", d["video"][va]),
    ("video_last_frame", d["video_last"][va]),
    ("joint_video_proprio", np.concatenate([d["video"][va], d["state"][va]], axis=1)),
):
    arm = reloaded_checkpoint["arms"][name]
    in_features = arm["state_dict"]["weight"].shape[1]
    reloaded_model = nn.Linear(in_features, 8)
    reloaded_model.load_state_dict(arm["state_dict"])
    with torch.inference_mode():
        reloaded_prob = (
            reloaded_model(
                (torch.tensor(source_features, dtype=torch.float32) - arm["feature_mean_train"])
                / arm["feature_std_train"]
            )
            .softmax(1)
            .numpy()
        )
    reload_max_abs_diff[name] = float(np.max(np.abs(reloaded_prob - predictions[name])))
if any(value > 1e-7 for value in reload_max_abs_diff.values()):
    raise RuntimeError(f"saved probe reload changed probabilities: {reload_max_abs_diff}")
results["checkpoint_reload_max_abs_probability_delta"] = reload_max_abs_diff
results["saved_predictions"] = {
    "path": str(predictions_path),
    "sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    "count": len(validation_predictions),
}
results["saved_probe_checkpoint"] = {
    "path": str(root / "probe-checkpoints.pt"),
    "sha256": hashlib.sha256((root / "probe-checkpoints.pt").read_bytes()).hexdigest(),
}
(root / "pilot-metrics.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
