"""Reusable native transcript-choice inference with an injected frozen Whisper."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from tiny_omni_decision.classification_json import classification_json
from tiny_omni_decision.transcript_likelihood import mean_transcript_logprob


@dataclass(frozen=True)
class EncodedSpeech:
    owner: object
    values: torch.Tensor


class NativeSpeechRuntime:
    """Fixed English transcript likelihood, not general acoustic reasoning.

    The caller owns/loading-verifies the model. No hidden model copy or tokenizer
    download. One encoded observation can serve multiple candidate sets.
    """

    def __init__(self, model, tokenizer, processor):
        if model.training:
            raise ValueError("inference model must already be in eval mode")
        if tokenizer.prefix_tokens != [50258, 50259, 50359, 50363]:
            raise ValueError("fixed English transcription prefix required")
        self.model, self.tokenizer, self.processor = model, tokenizer, processor
        self._owner = object()

    def encode(self, pcm, *, sample_rate):
        pcm = np.asarray(pcm)
        if (
            sample_rate != 16000
            or pcm.dtype != np.int16
            or pcm.ndim != 1
            or not 0 < len(pcm) <= self.processor.n_samples
        ):
            raise ValueError("mono 16kHz int16 PCM within the fixed window required")
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            features = self.processor(
                pcm.astype(np.float32) / 32768, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(device)
            values = self.model.model.encoder(features).last_hidden_state
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite acoustic encoding")
        return EncodedSpeech(self._owner, values)

    def scores(self, encoded, transcripts):
        if not isinstance(encoded, EncodedSpeech) or encoded.owner is not self._owner:
            raise ValueError("encoding must belong to this runtime")
        if (
            not 1 <= len(transcripts) <= 16
            or len(set(transcripts)) != len(transcripts)
            or any(not isinstance(s, str) or not s.strip() for s in transcripts)
        ):
            raise ValueError("1..16 distinct nonempty transcripts required")
        prefix = self.tokenizer.prefix_tokens
        sequences = [self.tokenizer(s)["input_ids"] for s in transcripts]
        for ids in sequences:
            if (
                ids[: len(prefix)] != prefix
                or ids[-1] != self.tokenizer.eos_token_id
                or len(ids) <= len(prefix)
                or len(ids) > self.model.config.max_target_positions
            ):
                raise ValueError("invalid or truncating candidate tokenization")
        lengths = [len(ids) for ids in sequences]
        device = encoded.values.device
        ids = torch.full(
            (len(transcripts), max(lengths)),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
            mask[i, : len(sequence)] = 1
        with torch.inference_mode():
            logits = self.model(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoded.values.expand(len(transcripts), -1, -1)
                ),
                decoder_input_ids=ids[:, :-1],
                decoder_attention_mask=mask[:, :-1],
                use_cache=False,
            ).logits
            chosen = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        values = mean_transcript_logprob(chosen.cpu().numpy(), lengths, prefix_length=len(prefix))
        return values, [n - len(prefix) for n in lengths]

    def decide(self, encoded, options):
        if not 2 <= len(options) <= 16 or any(
            set(o) != {"option_id", "transcript"} for o in options
        ):
            raise ValueError("2..16 option_id/transcript options required")
        transcripts = [o["transcript"] for o in options]
        # Validate IDs and duplicate transcripts before expensive decoding.
        supplied = [dict(option_id=o["option_id"], class_id=o["transcript"]) for o in options]
        classification_json([1 / len(options)] * len(options), transcripts, supplied)
        scores, counts = self.scores(encoded, transcripts)
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        result = classification_json(probabilities, transcripts, supplied)
        return dict(decision=result, scores=scores.tolist(), scored_token_counts=counts)
