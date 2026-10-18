import hashlib

import numpy as np
import pytest


def module():
    from scripts import native_video_cache
    return native_video_cache


def fixtures():
    rows = {s: [dict(media_sha256=c * 64, video_filename=f"video_{i}.mp4",
                     media_path=f"media/{s}/video_{i}.mp4", split=s, media_bytes=100)]
            for i, (s, c) in enumerate([("train", "a"), ("validation", "b")])}
    order = ["a" * 64, "b" * 64]
    prior = [dict(media_sha256=k, token_sha256="c" * 64) for k in order]
    return rows, order, prior


def test_complete_record_order_and_scene_binding():
    rows, order, prior = fixtures()
    records = module().records_for_cache(rows, order[::-1], prior)
    assert [r["media_sha256"] for r in records] == order[::-1]
    assert [r["split"] for r in records] == ["validation", "train"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "prior_missing"])
def test_never_silently_shrinks_or_adds_corpus(change):
    rows, order, prior = fixtures()
    if change == "missing":
        order.pop()
    if change == "duplicate":
        order.append(order[0])
    if change == "unknown":
        order[0] = "d" * 64
    if change == "prior_missing":
        prior.pop()
    with pytest.raises(ValueError):
        module().records_for_cache(rows, order, prior)


@pytest.mark.parametrize("key", ["media_sha256", "video_filename", "media_path"])
def test_underlying_identity_cannot_cross_split(key):
    rows, order, prior = fixtures()
    rows["validation"][0][key] = rows["train"][0][key]
    with pytest.raises(ValueError):
        module().records_for_cache(rows, order, prior)


def test_only_declared_development_splits_and_consistent_media():
    rows, order, prior = fixtures()
    rows["calibration"] = []
    with pytest.raises(ValueError):
        module().records_for_cache(rows, order, prior)
    rows.pop("calibration")
    rows["train"].append({**rows["train"][0], "media_bytes": 200})
    with pytest.raises(ValueError):
        module().records_for_cache(rows, order, prior)


def test_exact_fp32_tokens_saved_and_reloaded_without_overwrite(tmp_path):
    tokens = np.zeros((8, 16, 16, 768), dtype=np.float32)
    row = dict(media_sha256="a" * 64,
               token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest())
    record = module().store_tokens(tmp_path, row, tokens)
    assert record["shape"] == [2048, 768]
    assert record["reloaded_exact"]
    assert np.array_equal(np.load(tmp_path / record["file"]), tokens.reshape(2048, 768))
    with pytest.raises(FileExistsError):
        module().store_tokens(tmp_path, row, tokens)


@pytest.mark.parametrize("change", ["dtype", "shape", "nan", "hash"])
def test_invalid_tokens_never_get_published(tmp_path, change):
    tokens = np.zeros((8, 16, 16, 768), dtype=np.float32)
    expected = hashlib.sha256(tokens.tobytes()).hexdigest()
    if change == "dtype":
        tokens = tokens.astype(np.float16)
    if change == "shape":
        tokens = tokens[:4]
    if change == "nan":
        tokens.flat[0] = np.nan
    if change == "hash":
        tokens.flat[0] = 1
    with pytest.raises(ValueError):
        module().store_tokens(tmp_path, {"media_sha256": "a" * 64,
                                        "token_sha256": expected}, tokens)
    assert not list(tmp_path.iterdir())
