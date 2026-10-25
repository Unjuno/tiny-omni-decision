from pathlib import Path

import pytest

from scripts.path_a_videoprism_worker import _decode_image


class _FakeIO:
    def __init__(self):
        self.decoded = None

    @staticmethod
    def read_file(_path):
        return b"image-bytes"

    def decode_png(self, payload, *, channels):
        self.decoded = ("png", payload, channels)
        return self.decoded

    def decode_jpeg(self, payload, *, channels):
        self.decoded = ("jpeg", payload, channels)
        return self.decoded


class _FakeTF:
    def __init__(self):
        self.io = _FakeIO()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("image.png", "png"), ("image.jpg", "jpeg"), ("image.jpeg", "jpeg")],
)
def test_decode_image_selects_explicit_format(tmp_path: Path, name, expected):
    tf = _FakeTF()

    result = _decode_image(tf, tmp_path / name)

    assert result[0] == expected
    assert result[1:] == (b"image-bytes", 3)


def test_decode_image_rejects_unknown_extensions(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported image extension"):
        _decode_image(_FakeTF(), tmp_path / "image.gif")
