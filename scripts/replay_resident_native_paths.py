"""Concurrent resident visual/speech/sensor replay; no cross-modal fusion claim."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from scripts.evaluate_librispeech_audio_hard_negatives import CONDITIONS
from scripts.evaluate_native_speech import INPUT_HASH
from scripts.range_generalization import sha
from scripts.replay_shared_visual_decisions import replay
from tiny_omni_decision.native_sensor import ACTIVITY_CODES, CHANNELS
from tiny_omni_decision.native_sensor_runtime import StaticSensor, decide_sensor_window
from tiny_omni_decision.native_speech_runtime import NativeSpeechRuntime
from tiny_omni_decision.observations import ObservationBatch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Companion:
    def __init__(self):
        self.root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
        old = read(self.root / "native-speech-eval-01/config.json")
        self.model_path = Path(old["argv"]["model"])
        self.source = Path(old["argv"]["source"])
        examples = Path(old["argv"]["examples"])
        self.pins = {self.model_path / k: v for k, v in old["model_files"].items()}
        self.pins[examples] = INPUT_HASH
        sensor_weights = self.root / "sensor-subject-training-01/static/best.pt"
        self.pins[sensor_weights] = (
            "5780b04c5022de2f6a085c27032de05c5f86e84a6c638b02a93ed91a9b7db40d"
        )
        for path, digest in self.pins.items():
            if sha(path) != digest:
                raise ValueError(f"input hash mismatch: {path}")
        self.rows = [
            json.loads(line) for line in examples.read_text(encoding="utf-8").splitlines()
        ][:8]
        self.expected_speech = read(self.root / "native-speech-json-replay-03/decisions.json")
        self.sensor_payload = self.root / "native-sensor-json-replay-01/input-only.npz"
        self.expected_sensor = read(self.root / "native-sensor-json-replay-01/result.json")
        self.pins[self.sensor_payload] = sha(self.sensor_payload)
        self.speech = (
            WhisperForConditionalGeneration.from_pretrained(self.model_path, local_files_only=True)
            .eval()
            .to("cuda")
        )
        tokenizer = WhisperTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            language="english",
            task="transcribe",
            predict_timestamps=False,
        )
        processor = WhisperFeatureExtractor.from_pretrained(self.model_path, local_files_only=True)
        self.runtime = NativeSpeechRuntime(self.speech, tokenizer, processor)
        self.sensor = StaticSensor().eval()
        self.sensor.load_state_dict(torch.load(sensor_weights, weights_only=True))
        self.parameters = sum(p.numel() for m in (self.speech, self.sensor) for p in m.parameters())

    def run(self):
        started = time.perf_counter()
        versions = [p._version for m in (self.speech, self.sensor) for p in m.parameters()]
        torch.cuda.reset_peak_memory_stats()
        outputs = []
        speech_delta = sensor_delta = 0.0
        expected = {(r["id"], r["condition"]): r["output"] for r in self.expected_speech}
        for row in self.rows:
            audio = self.source / row["audio_path"]
            if sha(audio) != row["audio_sha256"]:
                raise ValueError("audio changed")
            pcm, rate = sf.read(audio, dtype="int16")
            encoded = self.runtime.encode(pcm, sample_rate=rate)
            sets = [
                [
                    dict(option_id=f"candidate-{i}", transcript=t)
                    for i, t in enumerate(row["options_by_condition"][condition])
                ]
                for condition in CONDITIONS
            ]
            results = self.runtime.decide_many(encoded, sets)
            for condition, result in zip(CONDITIONS, results, strict=True):
                reference = expected[row["id"], condition]
                p = np.array([o["probability"] for o in result["decision"]["options"]])
                q = np.array([o["probability"] for o in reference["decision"]["options"]])
                np.testing.assert_allclose(p, q, atol=1e-4, rtol=0)
                np.testing.assert_allclose(result["scores"], reference["scores"], atol=1e-4, rtol=0)
                assert (
                    result["decision"]["selected_option_id"]
                    == reference["decision"]["selected_option_id"]
                )
                speech_delta = max(speech_delta, float(abs(p - q).max()))
                outputs.append(
                    dict(modality="audio", id=row["id"], condition=condition, result=result)
                )
        with np.load(self.sensor_payload, allow_pickle=False) as data:
            for i, reference in enumerate(self.expected_sensor["decisions"]):
                times = data["timestamps_ms"][i].tolist()
                clock = reference["decision"]["temporal_scope"]["clock_id"]
                batch = ObservationBatch.model_validate(
                    dict(
                        schema_version="1.0",
                        observations=[
                            dict(
                                modality="sensor",
                                source_id=reference["sample_id"],
                                payload_ref=str(self.sensor_payload),
                                payload_index=i,
                                payload_sha256=self.pins[self.sensor_payload],
                                clock_id=clock,
                                timestamps_ms=times,
                                valid_mask=[True] * 128,
                                channels=[
                                    dict(name=c, unit="g", scale=1.0, offset=0.0) for c in CHANNELS
                                ],
                            )
                        ],
                        options=[
                            dict(option_id=f"custom-{c}", attributes={"activity_code": c})
                            for c in ACTIVITY_CODES
                        ],
                    )
                )
                result = decide_sensor_window(
                    self.sensor, batch, query_time_ms=times[-1], clock_id=clock
                )
                p = np.array([o["probability"] for o in result["options"]])
                q = np.array([o["probability"] for o in reference["decision"]["options"]])
                np.testing.assert_allclose(p, q, atol=1e-5, rtol=0)
                assert result["selected_option_id"] == reference["decision"]["selected_option_id"]
                sensor_delta = max(sensor_delta, float(abs(p - q).max()))
                outputs.append(dict(modality="sensor", id=reference["sample_id"], result=result))
        assert versions == [p._version for m in (self.speech, self.sensor) for p in m.parameters()]
        assert all(sha(path) == digest for path, digest in self.pins.items())
        return dict(
            status="PASS_RESIDENT_COMPANIONS",
            outputs=outputs,
            parameters=self.parameters,
            max_speech_probability_delta=speech_delta,
            max_sensor_probability_delta=sensor_delta,
            changed_answers=0,
            seconds=time.perf_counter() - started,
            host_memory=psutil.Process().memory_info()._asdict(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            pins={str(p): h for p, h in self.pins.items()},
            scope=(
                "Concurrent residency and independent decisions; "
                "no joint task or standalone text decision."
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    torch.set_num_threads(2)
    companion = Companion()
    replay(args.output, companion=companion)


if __name__ == "__main__":
    main()
