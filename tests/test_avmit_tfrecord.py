from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from tiny_omni_decision.avmit_tfrecord import (
    _masked_crc32c,
    crc32c,
    decode_float32_steps,
    iter_tfrecord_payloads,
    parse_avmit_sequence_example,
)


def _varint(value: int) -> bytes:
    output = bytearray()
    while value > 0x7F:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _feature(value: bytes) -> bytes:
    return _bytes_field(1, _bytes_field(1, value))


def _map_entry(key: str, value: bytes) -> bytes:
    return _bytes_field(1, key.encode()) + _bytes_field(2, value)


def _sequence_example() -> bytes:
    context = _bytes_field(1, _map_entry("filename", _feature(b"class/source.mp4")))
    context += _bytes_field(1, _map_entry("label", _feature(b"must-not-be-returned")))
    audio_values = [
        np.array([1.25, -3.0], dtype="<f4").tobytes(),
        np.array([2.0, 4.5], dtype="<f4").tobytes(),
    ]
    video_values = [np.array([0.5], dtype="<f4").tobytes()]
    feature_lists = _bytes_field(
        1,
        _map_entry(
            "audio",
            b"".join(_bytes_field(1, _feature(value)) for value in audio_values),
        ),
    ) + _bytes_field(
        1,
        _map_entry(
            "visual_object",
            b"".join(_bytes_field(1, _feature(value)) for value in video_values),
        ),
    )
    return _bytes_field(1, context) + _bytes_field(2, feature_lists)


def _frame(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", _masked_crc32c(length))
        + payload
        + struct.pack("<I", _masked_crc32c(payload))
    )


def test_crc32c_known_castagnoli_vector():
    assert crc32c(b"123456789") == 0xE3069283


def test_tfrecord_stream_validates_frame_and_payload_checksums():
    payload = b"sequence-example"
    assert list(iter_tfrecord_payloads(io.BytesIO(_frame(payload)))) == [payload]

    corrupt_length_crc = bytearray(_frame(payload))
    corrupt_length_crc[8] ^= 1
    with pytest.raises(ValueError, match="length checksum"):
        list(iter_tfrecord_payloads(io.BytesIO(corrupt_length_crc)))

    corrupt_payload_crc = bytearray(_frame(payload))
    corrupt_payload_crc[-1] ^= 1
    with pytest.raises(ValueError, match="payload checksum"):
        list(iter_tfrecord_payloads(io.BytesIO(corrupt_payload_crc)))

    with pytest.raises(ValueError, match="truncated TFRecord"):
        list(iter_tfrecord_payloads(io.BytesIO(b"\x01")))


def test_avmit_sequence_parser_reads_only_filename_and_requested_features():
    parsed = parse_avmit_sequence_example(_sequence_example())

    assert parsed["filename"] == "class/source.mp4"
    assert set(parsed) == {"filename", "audio", "visual_object"}
    audio = decode_float32_steps(parsed["audio"])
    video = decode_float32_steps(parsed["visual_object"])
    np.testing.assert_array_equal(audio, [[1.25, -3.0], [2.0, 4.5]])
    np.testing.assert_array_equal(video, [[0.5]])


def test_avmit_sequence_parser_rejects_missing_streams_and_nonfinite_values():
    empty_sequences = _bytes_field(
        1, _bytes_field(1, _map_entry("filename", _feature(b"class/source.mp4")))
    ) + _bytes_field(2, b"")
    with pytest.raises(ValueError, match="audio and visual_object"):
        parse_avmit_sequence_example(empty_sequences)
    with pytest.raises(ValueError, match="finite"):
        decode_float32_steps([np.array([np.inf], dtype="<f4").tobytes()])
