"""Real raw image/video requests through one worker and existing decision heads."""

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from scripts.train_dinov2_image_probe import sha, write
from scripts.train_frozen_clevrer_binary_reasoning import _validate_inputs
from scripts.train_frozen_video_probe import _text_features
from tiny_omni_decision.classification_json import classification_json
from tiny_omni_decision.token_readout import VideoTokenReadout
from tiny_omni_decision.token_transport import read_tokens
from tiny_omni_decision.visual_requests import VisualRequest


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    out = root / "shared-visual-decision-replay-01"
    out.mkdir(exist_ok=False)
    config_path = Path("configs/pretrained_reuse/path_a_videoprism_clevrer_binary_reasoning.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checked = _validate_inputs(config_path, config)
    first_scene = checked["validation_rows"][0]["scene_index"]
    video_rows = [r for r in checked["validation_rows"] if r["scene_index"] == first_scene][:8]
    cache = root / "videoprism-spatial-image-cache-01"
    assert sha(cache / "records.json") == (
        "32944fc822c45a7a93ee06904c9a97acaa5bf6364c33c40525f535c59da55b41"
    )
    image_rows = read(cache / "records.json")["val"][:8]
    image_assets = [
        dict(
            path=str(root / "clevr4-sample-seed17-1024-512/val/images" / (r["id"] + ".png")),
            sha256=r["sha256"],
        )
        for r in image_rows
    ]
    video_asset = dict(
        path=str(Path(config["source"]["sample_dir"]) / video_rows[0]["media_path"]),
        sha256=video_rows[0]["media_sha256"],
    )
    requests = [
        VisualRequest(key="images", modality="image", assets=image_assets),
        VisualRequest(key="video", modality="video", assets=[video_asset]),
        VisualRequest(key="images-repeat", modality="image", assets=image_assets),
    ]
    write(out / "requests.json", [asdict(r) for r in requests])
    image_path = root / "shared-affine-regularized-01/0.01/final.pt"
    video_path = root / "video-token-readout-01/attention/best.pt"
    pins = {
        image_path: "f815496df64ec5e59000d08a22d42b4ecf2b7e78758df129a797321079b8b0d5",
        video_path: "ee4b46156e7e99149a1b8dca2701a786431cf09f0dbafbe9e247104ead16ba62",
    }
    for path, digest in pins.items():
        assert sha(path) == digest
    image_expected_path = root / "shared-affine-regularized-01/0.01/validation-predictions.json"
    video_expected_path = root / "video-token-readout-01/attention/predictions.json"
    image_expected = read(image_expected_path)[:32]
    saved_video = {r["id"]: r for r in read(video_expected_path)}
    video_expected = [saved_video[r["id"]] for r in video_rows]
    torch.set_num_threads(2)
    image_model = torch.nn.Linear(1536, 40).double().eval()
    image_model.load_state_dict(torch.load(image_path, weights_only=True))
    video_model = VideoTokenReadout().eval()
    video_model.load_state_dict(torch.load(video_path, weights_only=True))
    text_path = config["text_encoder"]["model_path"]
    text_model = AutoModel.from_pretrained(text_path, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
    features, _ = _text_features(video_rows, tokenizer, text_model, torch.device("cpu"))
    text = torch.stack(features)
    write(
        out / "config.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            request_sha256=sha(out / "requests.json"),
            weights={str(p): h for p, h in pins.items()},
            expected_hashes={str(p): sha(p) for p in (image_expected_path, video_expected_path)},
            probability_atol=1e-4,
            allowed_answer_changes=0,
            feature_atol=1e-5,
            question_ids=[r["id"] for r in video_rows],
            training=False,
            sealed_audit=False,
            cpu_threads=2,
        ),
    )
    python = Path(
        "C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe"
    )
    outputs, arrays = {}, {}
    with (out / "worker.log").open("wb") as log:
        with subprocess.Popen(
            [
                str(python),
                "-m",
                "scripts.run_shared_visual_requests",
                "--requests",
                str(out / "requests.json"),
                "--report",
                str(out / "worker-report.json"),
            ],
            stdout=subprocess.PIPE,
            stderr=log,
            env=dict(os.environ, PYTHONPATH=str(Path.cwd() / "src"), JAX_PLATFORMS="cpu"),
            creationflags=subprocess.CREATE_NO_WINDOW,
        ) as worker:
            for request in requests:
                result = read_tokens(worker.stdout, request.output_shape())
                if result is None or result[0] != request.key:
                    raise ValueError("worker output missing or out of order")
                values = result[1]
                arrays[request.key] = values
                np.save(out / f"{request.key}.npy", values, allow_pickle=False)
                with torch.inference_mode():
                    if request.modality == "image":
                        x = torch.from_numpy(
                            np.concatenate((values[:, 0], values[:, 1:].mean(1)), 1)
                        )
                        probabilities = image_model(x.double()).reshape(-1, 10).softmax(-1).numpy()
                        expected = image_expected
                    else:
                        tokens = torch.from_numpy(values.copy()).expand(len(video_rows), -1, -1)
                        probabilities = video_model(tokens, text).softmax(-1).numpy()
                        expected = video_expected
                reference = np.array([r["probabilities"] for r in expected])
                np.testing.assert_allclose(probabilities, reference, atol=1e-4, rtol=0)
                assert np.array_equal(probabilities.argmax(-1), reference.argmax(-1))
                decisions = []
                for index, (row, p) in enumerate(zip(expected, probabilities, strict=True)):
                    classes = row.get(
                        "options",
                        video_rows[index]["options"] if request.modality == "video" else [],
                    )
                    options = [
                        dict(option_id=f"custom-{i}", class_id=c) for i, c in enumerate(classes)
                    ]
                    a = classification_json(p, classes, options)
                    b = classification_json(p, classes, options[::-1])
                    assert a["selected_option_id"] == b["selected_option_id"]
                    decisions.append(dict(id=row["id"], decision=a))
                outputs[request.key] = dict(
                    decisions=decisions,
                    max_probability_delta=float(abs(probabilities - reference).max()),
                )
            if worker.stdout.read(1) or worker.wait(timeout=30) != 0:
                raise ValueError("unexpected worker termination/output")
    np.testing.assert_array_equal(arrays["images"], arrays["images-repeat"])
    np.testing.assert_allclose(
        arrays["images"], np.load(cache / "val-cells.npy")[:8], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        arrays["video"].mean(1)[0],
        checked["feature_by_hash"][video_asset["sha256"]],
        atol=1e-5,
        rtol=0,
    )
    assert all(sha(p) == h for p, h in pins.items())
    write(out / "decisions.json", outputs)
    write(
        out / "result.json",
        dict(
            status="PASS_RAW_SHARED_DECISIONS",
            outputs=outputs,
            worker=read(out / "worker-report.json"),
            changed_answers=0,
            exact_image_repeat=True,
            source_unchanged=True,
            full_omni_runtime=False,
            parameters=114365184
            + sum(
                p.numel() for m in (image_model, video_model, text_model) for p in m.parameters()
            ),
            hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    print("PASS_RAW_SHARED_DECISIONS")


if __name__ == "__main__":
    main()
