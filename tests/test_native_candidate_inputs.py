import pytest

torch = pytest.importorskip("torch")


def test_positions_preserve_temporal_and_spatial_grid():
    from scripts.native_candidate_inputs import positions

    video = positions("video", 2048)
    assert video.shape == (2048, 3)
    assert torch.equal(video[0], torch.zeros(3))
    assert torch.equal(video[-1], torch.ones(3))
    assert torch.equal(video[256], torch.tensor([1 / 7, 0, 0]))
    image = positions("image", 17)
    assert torch.equal(image[0], torch.tensor([0., .5, .5]))
    assert torch.equal(image[1], torch.zeros(3))
    assert torch.equal(image[-1], torch.tensor([0., 1., 1.]))
    speech = positions("speech", 110)
    assert torch.equal(speech[:, 1:], torch.zeros(110, 2))
    assert speech[0, 0] == 0 and speech[-1, 0] == 1
    for kind, count in [("image", 16), ("video", 4096), ("speech", 0), ("bad", 2)]:
        with pytest.raises(ValueError):
            positions(kind, count)


def test_media_gate_requires_declared_train_assets_and_hashes(tmp_path):
    from scripts.native_candidate_inputs import validate_native_sources

    image = {"train": [{"id": "i", "sha256": "a" * 64, "split": "train"}]}
    av = [dict(source_id="s", split="train", audio_sha256="b" * 64,
               video_sha256="c" * 64, audio_status="PASS", video_status="PASS")]
    validate_native_sources(image, av, image_id="i", image_sha="a" * 64,
                            source_id="s", audio_sha="b" * 64, video_sha="c" * 64)
    for key in ["split", "audio_sha256", "video_sha256", "audio_status", "video_status"]:
        changed = [dict(av[0], **{key: "wrong"})]
        with pytest.raises(ValueError):
            validate_native_sources(image, changed, image_id="i", image_sha="a" * 64,
                                    source_id="s", audio_sha="b" * 64, video_sha="c" * 64)


def test_masked_language_pool_rejects_truncation_and_nonfinite():
    from scripts.native_candidate_inputs import masked_mean

    x = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]])
    assert torch.equal(masked_mean(x, torch.tensor([[1, 1, 0]])), torch.tensor([[2., 3.]]))
    for mask in [torch.zeros(1, 3), torch.ones(1, 2)]:
        with pytest.raises(ValueError):
            masked_mean(x, mask)
