import hashlib

import pytest

pytest.importorskip("torch")
from scripts.build_source_semantics_review_sheet import blinded_rows, render  # noqa: E402


def test_rows_are_blinded_and_order_independent(tmp_path):
    samples = []
    for index in range(16):
        path = tmp_path / f"video_{index}.mp4"
        path.write_bytes(f"clip{index}".encode())
        samples.append(
            {
                "record_id": f"video_{index}:sound:1",
                "category": "hidden-label",
                "media_path": str(path),
                "media_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "audio_interval_us": [1000000, 2000000],
            }
        )
    rows = blinded_rows({"samples": samples})
    assert rows == blinded_rows({"samples": list(reversed(samples))})
    assert len(rows) == 16 and all("category" not in row for row in rows)
    rendered = render(rows)
    assert "hidden-label" not in rendered and "source-semantics-review.json" in rendered
    (tmp_path / "video_1.mp4").write_bytes(b"altered")
    with pytest.raises(ValueError, match="media hash"):
        blinded_rows({"samples": samples})
