from types import SimpleNamespace

import numpy as np
import pytest


def make_runtime():
    torch = pytest.importorskip("torch")
    from tiny_omni_decision.native_speech_runtime import NativeSpeechRuntime

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0
            self.decodes = 0
            self.config = SimpleNamespace(max_target_positions=8)
            self.model = SimpleNamespace(encoder=self.encode)

        def encode(self, features):
            self.calls += 1
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 2, 3))

        def forward(self, **kwargs):
            self.decodes += 1
            shape = kwargs["decoder_input_ids"].shape
            return SimpleNamespace(logits=torch.zeros(*shape, 50364))

    class Tokenizer:
        prefix_tokens = [50258, 50259, 50359, 50363]
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, text):
            return {"input_ids": self.prefix_tokens + [2] * len(text) + [1]}

    class Processor:
        n_samples = 100

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

    return NativeSpeechRuntime(Model().eval(), Tokenizer(), Processor())


def test_encode_once_and_reuse_without_changing_weights():
    runtime = make_runtime()
    encoded = runtime.encode(np.zeros(32, np.int16), sample_rate=16000)
    options = [dict(option_id="one", transcript="a"), dict(option_id="two", transcript="bb")]
    a = runtime.decide(encoded, options)
    b = runtime.decide(encoded, options[::-1])
    assert runtime.model.calls == 1
    assert a["decision"]["selected_option_id"] == b["decision"]["selected_option_id"]
    assert a["scored_token_counts"] == [2, 3]
    assert runtime.model.weight.item() == 0
    assert runtime.model.weight.grad is None


def test_invalid_audio_ownership_and_truncation_fail_closed():
    runtime = make_runtime()
    with pytest.raises(ValueError):
        runtime.encode(np.zeros(32, np.float32), sample_rate=16000)
    with pytest.raises(ValueError):
        runtime.encode(np.zeros(101, np.int16), sample_rate=16000)
    encoded = runtime.encode(np.zeros(32, np.int16), sample_rate=16000)
    with pytest.raises(ValueError):
        make_runtime().scores(encoded, ["a"])
    with pytest.raises(ValueError):
        runtime.scores(encoded, ["much too long"])
    with pytest.raises(ValueError):
        runtime.decide(encoded, [dict(option_id="same", transcript=s) for s in ("a", "b")])


def test_many_candidate_sets_share_one_decoder_batch():
    runtime = make_runtime()
    encoded = runtime.encode(np.zeros(32, np.int16), sample_rate=16000)
    sets = [[dict(option_id=str(i), transcript=s) for i, s in enumerate(values)]
            for values in (("a", "bb"), ("bb", "c"))]
    results = runtime.decide_many(encoded, sets)
    assert len(results) == 2
    assert runtime.model.calls == 1 and runtime.model.decodes == 1
    assert results[0]["scores"][1] == results[1]["scores"][0]
