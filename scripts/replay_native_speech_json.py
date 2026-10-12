"""Eight fixed real audio assets, reused encodings and custom-ID JSON parity."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

from scripts.evaluate_librispeech_audio_hard_negatives import CONDITIONS
from scripts.evaluate_native_speech import INPUT_HASH
from scripts.range_generalization import sha, write
from tiny_omni_decision.native_speech_runtime import NativeSpeechRuntime


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    previous = root / "native-speech-eval-01"
    old_config = json.loads((previous / "config.json").read_text(encoding="utf-8"))
    examples = Path(old_config["argv"]["examples"])
    model_path = Path(old_config["argv"]["model"])
    source = Path(old_config["argv"]["source"])
    assert sha(examples) == INPUT_HASH
    for name, digest in old_config["model_files"].items():
        assert sha(model_path / name) == digest
    rows = [json.loads(line) for line in examples.read_text(encoding="utf-8").splitlines()][:8]
    expected = {
        condition: [
            json.loads(line)
            for line in (previous / f"{condition}-predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ][:8]
        for condition in CONDITIONS
    }
    out = root / "native-speech-json-replay-01"
    out.mkdir(exist_ok=False)
    write(
        out / "config.json",
        dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            model_files=old_config["model_files"],
            input_sha256=INPUT_HASH,
            sample_ids=[r["id"] for r in rows],
            device="cpu",
            dtype="float32",
            threads=2,
            probability_atol=1e-4,
            score_atol=1e-4,
            allowed_answer_changes=0,
            expected_hashes={
                condition: sha(previous / f"{condition}-predictions.jsonl")
                for condition in CONDITIONS
            },
            training=False,
            sealed_audit=False,
        ),
    )
    torch.set_num_threads(2)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True
    ).eval()
    tokenizer = WhisperTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        language="english",
        task="transcribe",
        predict_timestamps=False,
    )
    processor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    runtime = NativeSpeechRuntime(model, tokenizer, processor)
    versions = {name: p._version for name, p in model.named_parameters()}
    encodes = []
    hook = model.model.encoder.register_forward_hook(lambda *args: encodes.append(1))
    records = []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        audio = source / row["audio_path"]
        assert sha(audio) == row["audio_sha256"]
        pcm, rate = sf.read(audio, dtype="int16")
        tick = time.perf_counter()
        encoded = runtime.encode(pcm, sample_rate=rate)
        encode_seconds = time.perf_counter() - tick
        for condition in CONDITIONS:
            reference = expected[condition][i]
            choices = row["options_by_condition"][condition]
            assert reference["sample_id"] == row["id"] and reference["options"] == choices
            assert reference["audio_sha256"] == row["audio_sha256"]
            options = [
                dict(option_id=f"candidate-{j}", transcript=s) for j, s in enumerate(choices)
            ]
            tick = time.perf_counter()
            output = runtime.decide(encoded, options)
            score_seconds = time.perf_counter() - tick
            reverse = runtime.decide(encoded, options[::-1])
            a, b = output["decision"], reverse["decision"]
            assert a["selected_option_id"] == b["selected_option_id"]
            probabilities = [o["probability"] for o in a["options"]]
            np.testing.assert_allclose(probabilities, reference["probabilities"], atol=1e-4, rtol=0)
            np.testing.assert_allclose(output["scores"], reference["scores"], atol=1e-4, rtol=0)
            np.testing.assert_allclose(
                probabilities, [o["probability"] for o in b["options"]][::-1], atol=1e-5, rtol=0
            )
            assert a["selected_option_id"] == f"candidate-{reference['prediction_index']}"
            assert output["scored_token_counts"] == reference["scored_token_counts"]
            records.append(
                dict(
                    id=row["id"],
                    condition=condition,
                    output=output,
                    encode_seconds=encode_seconds,
                    score_seconds=score_seconds,
                    audio_sha256=row["audio_sha256"],
                    encoding_sha256=hashlib.sha256(
                        encoded.values.cpu().numpy().tobytes()
                    ).hexdigest(),
                    max_probability_delta=float(
                        np.max(np.abs(np.array(probabilities) - reference["probabilities"]))
                    ),
                    max_score_delta=float(
                        np.max(np.abs(np.array(output["scores"]) - reference["scores"]))
                    ),
                )
            )
        print(f"completed {i + 1}/8 audio assets", flush=True)
    hook.remove()
    assert len(encodes) == 8
    assert versions == {name: p._version for name, p in model.named_parameters()}
    for name, digest in old_config["model_files"].items():
        assert sha(model_path / name) == digest
    write(out / "decisions.json", records)
    write(
        out / "result.json",
        dict(
            status="PASS_NATIVE_SPEECH_JSON",
            assets=8,
            decisions=16,
            encoder_calls=len(encodes),
            decoder_candidate_sets=32,
            changed_answers=0,
            parameters=sum(p.numel() for p in model.parameters()),
            seconds=time.perf_counter() - started,
            max_probability_delta=max(r["max_probability_delta"] for r in records),
            max_score_delta=max(r["max_score_delta"] for r in records),
            weights_unchanged=True,
            candidate_order_stable=True,
            config_sha256=sha(out / "config.json"),
            decisions_sha256=sha(out / "decisions.json"),
        ),
    )


if __name__ == "__main__":
    main()
