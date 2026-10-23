from __future__ import annotations

import hashlib
import io

import pytest

from scripts.preflight_ami_rm_media import (
    REQUIRED_MEETINGS,
    build_stream_url,
    parse_directory_index,
    write_verified_stream,
)


def _index(rows: list[tuple[str, str]]) -> str:
    return "".join(
        f"<tr><td></td><td><a href=\"{name}\">{name}</a></td>"
        "<td align=\"right\">2005-01-01</td>"
        f"<td align=\"right\">{size}</td><td></td></tr>"
        for name, size in rows
    )


def test_parse_directory_index_requires_exact_four_camera_rm_files() -> None:
    html = _index(
        [
            (f"ES2010b.Closeup{camera}.rm", f"{camera + 10}M")
            for camera in range(1, 5)
        ]
        + [("ES2010b.Closeup1.avi", "400M"), ("other.rm", "10M")]
    )

    entries = parse_directory_index(html, "ES2010b")

    assert set(entries) == {1, 2, 3, 4}
    assert entries[1].name == "ES2010b.Closeup1.rm"
    assert entries[1].size_label == "11M"


def test_parse_directory_index_rejects_missing_or_duplicate_camera() -> None:
    with pytest.raises(ValueError, match="all four"):
        parse_directory_index(
            _index([(f"ES2010b.Closeup{i}.rm", "10M") for i in range(1, 4)]),
            "ES2010b",
        )

    duplicate = _index(
        [(f"ES2010b.Closeup{i}.rm", "10M") for i in range(1, 5)]
        + [("ES2010b.Closeup1.rm", "10M")]
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_directory_index(duplicate, "ES2010b")


@pytest.mark.parametrize(
    "href",
    [
        "../ES2010b.Closeup1.rm",
        "/outside.rm",
        "https://evil.invalid/a.rm",
    ],
)
def test_parse_directory_index_rejects_nonlocal_media_links(href: str) -> None:
    html = _index([(f"ES2010b.Closeup{i}.rm", "10M") for i in range(2, 5)])
    html += (
        f"<tr><td></td><td><a href=\"{href}\">ES2010b.Closeup1.rm</a></td>"
        "<td>date</td><td>10M</td><td></td></tr>"
    )

    with pytest.raises(ValueError, match="unsafe"):
        parse_directory_index(html, "ES2010b")


def test_build_stream_url_is_limited_to_pinned_cohort_and_camera() -> None:
    assert build_stream_url("ES2010b", 4).endswith(
        "/ES2010b/video/ES2010b.Closeup4.rm"
    )
    with pytest.raises(ValueError, match="meeting"):
        build_stream_url("ES2002a", 1)
    with pytest.raises(ValueError, match="camera"):
        build_stream_url("ES2010b", 5)


def test_write_verified_stream_hashes_and_refuses_overwrite(tmp_path) -> None:
    payload = b"fixed media bytes"
    destination = tmp_path / "camera.rm"

    digest = write_verified_stream(io.BytesIO(payload), destination, len(payload))

    assert destination.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        write_verified_stream(io.BytesIO(payload), destination, len(payload))


def test_write_verified_stream_rejects_length_mismatch_without_final_file(tmp_path) -> None:
    destination = tmp_path / "camera.rm"

    with pytest.raises(ValueError, match="byte count"):
        write_verified_stream(io.BytesIO(b"short"), destination, 10)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_pinned_meeting_set_has_three_groups_per_split() -> None:
    assert len(REQUIRED_MEETINGS["train"]) == 3
    assert len(REQUIRED_MEETINGS["validation"]) == 3
    assert set(REQUIRED_MEETINGS["train"]).isdisjoint(REQUIRED_MEETINGS["validation"])
