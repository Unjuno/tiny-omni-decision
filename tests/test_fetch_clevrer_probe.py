import pytest

torch = pytest.importorskip("torch")  # noqa: F401
np = pytest.importorskip("numpy")
pytest.importorskip("transformers")
pytest.importorskip("av")

from scripts.fetch_clevrer_probe import RangeFile, choose, choose_validation  # noqa: E402
from scripts.train_frozen_video_probe import _uniform_indices, _video_features  # noqa: E402


def _question(question_id: int, text: str, program: list[str]) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": text,
        "question_type": "descriptive",
        "question_subtype": "exist",
        "program": program,
        "answer": "yes",
    }


def _rows() -> list[dict[str, object]]:
    return [
        {
            "scene_index": scene,
            "video_filename": f"video_{scene:05d}.mp4",
            "questions": [
                _question(0, f"Are objects visible in scene {scene}?", ["objects", "exist"]),
                _question(
                    1,
                    f"What happens before collision in scene {scene}?",
                    ["filter_collision", "before"],
                ),
            ],
        }
        for scene in range(8)
    ]


def test_clevrer_probe_is_deterministic_scene_grouped_and_excludes_old_scenes():
    selected = choose(_rows(), "train", per_type=2, seed=17)
    repeated = choose(_rows(), "train", per_type=2, seed=17)

    assert [(row["scene_index"], row["question_id"]) for row in selected] == [
        (row["scene_index"], row["question_id"]) for row in repeated
    ]
    scenes = {row["scene_index"] for row in selected}
    assert len(scenes) == 2
    assert not ({0, 1, 2} & {row["scene_index"] for row in selected})
    assert {row["question_type"] for row in selected} == {
        "static_descriptive",
        "temporal_descriptive",
    }
    for scene in scenes:
        assert {row["question_type"] for row in selected if row["scene_index"] == scene} == {
            "temporal_descriptive",
            "static_descriptive",
        }


def test_validation_scene_selector_excludes_train_question_content():
    train = choose(_rows(), "train", per_type=2, seed=17)
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    validation = choose(_rows(), "validation", per_type=2, seed=17, excluded_content=train_content)
    validation_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in validation
    }

    assert not train_content & validation_content


def test_validation_template_overlap_can_be_allowed_while_scene_groups_stay_disjoint():
    train = choose(_rows(), "train", per_type=2, seed=17)
    train_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in train
    }
    validation_rows = [
        {**row, "scene_index": int(row["scene_index"]) + 10_000}
        for row in _rows()
    ]

    validation = choose_validation(
        validation_rows,
        per_type=2,
        seed=17,
        train_content=train_content,
        excluded_scenes=set(),
        allow_question_template_overlap=True,
    )

    assert {row["scene_index"] for row in validation}.isdisjoint(
        {row["scene_index"] for row in train}
    )
    validation_content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"])))
        for row in validation
    }
    assert train_content & validation_content


def test_scene_selector_excludes_previously_consumed_scenes_deterministically():
    excluded = {3, 4, 5}
    selected = choose(
        _rows(), "train", per_type=2, seed=17, excluded_scenes=excluded
    )
    repeated = choose(
        _rows(), "train", per_type=2, seed=17, excluded_scenes=excluded
    )

    scenes = {row["scene_index"] for row in selected}
    assert scenes == {row["scene_index"] for row in repeated}
    assert scenes.isdisjoint(excluded)
    assert scenes.isdisjoint({0, 1, 2})


def test_range_file_read_ahead_reuses_verified_http_range_buffer(monkeypatch):
    import scripts.fetch_clevrer_probe as fetch_probe

    content = bytes(range(256)) * 8192
    etag = '"fixture-etag"'

    class Response:
        def __init__(self, *, data=b"", headers=None, status_code=206):
            self.content = data
            self.headers = headers or {}
            self.status_code = status_code

        def raise_for_status(self):
            return None

    class Session:
        def __init__(self):
            self.requests = []

        def head(self, _url, **_kwargs):
            return Response(
                headers={"Content-Length": str(len(content)), "ETag": etag},
                status_code=200,
            )

        def get(self, _url, *, headers, **_kwargs):
            start, end = [int(value) for value in headers["Range"][len("bytes=") :].split("-")]
            self.requests.append((start, end))
            return Response(
                data=content[start : end + 1],
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(content)}",
                    "ETag": etag,
                },
            )

    session = Session()
    monkeypatch.setattr(fetch_probe.requests, "Session", lambda: session)
    archive = RangeFile("https://example.test/archive.zip", len(content), etag)

    assert archive.read(32) == content[:32]
    assert archive.read(100) == content[32:132]
    archive.seek(80)
    assert archive.read(16) == content[80:96]
    assert len(session.requests) == 1
    archive.seek(fetch_probe.RANGE_READ_AHEAD_BYTES + 10)
    assert archive.read(20) == content[fetch_probe.RANGE_READ_AHEAD_BYTES + 10 :][:20]
    assert len(session.requests) == 2
    assert archive.range_request_count == 2


def test_parallel_archive_chunk_extracts_only_hash_verified_selected_media(
    tmp_path, monkeypatch
):
    import io
    import zipfile

    import scripts.fetch_clevrer_probe as fetch_probe

    media = b"\x00\x00\x00\x18ftypisom" + b"fixture-mp4-data" * 16
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("video_00001.mp4", media)
    content = buffer.getvalue()
    etag = '"fixture-etag"'

    class Response:
        def __init__(self, *, data=b"", headers=None, status_code=206):
            self.content = data
            self.headers = headers or {}
            self.status_code = status_code

        def raise_for_status(self):
            return None

    class Session:
        def head(self, _url, **_kwargs):
            return Response(
                headers={"Content-Length": str(len(content)), "ETag": etag},
                status_code=200,
            )

        def get(self, _url, *, headers, **_kwargs):
            start, end = [int(value) for value in headers["Range"][len("bytes=") :].split("-")]
            return Response(
                data=content[start : end + 1],
                headers={
                    "Content-Range": f"bytes {start}-{end}/{len(content)}",
                    "ETag": etag,
                },
            )

    monkeypatch.setattr(fetch_probe.requests, "Session", Session)
    records = [{"scene_index": 1, "video_filename": "video_00001.mp4"}]
    extracted, range_bytes, requests, resumed = fetch_probe._extract_archive_chunk(
        records,
        {"url": "https://example.test/archive.zip", "bytes": len(content), "etag": etag},
        tmp_path,
        "train",
    )

    output = tmp_path / "media" / "train" / "video_00001.mp4"
    assert output.read_bytes() == media
    assert extracted == {1: ("video_00001.mp4", len(media), fetch_probe.sha256(media))}
    assert range_bytes > len(media)
    assert requests >= 1
    assert resumed == 0

    reused, _, _, resumed = fetch_probe._extract_archive_chunk(
        records,
        {"url": "https://example.test/archive.zip", "bytes": len(content), "etag": etag},
        tmp_path,
        "train",
        resume_partial=True,
    )
    assert reused == extracted
    assert resumed == 1
    output.write_bytes(b"wrong-content" * 3)
    with pytest.raises(ValueError, match="does not match pinned ZIP entry"):
        fetch_probe._extract_archive_chunk(
            records,
            {"url": "https://example.test/archive.zip", "bytes": len(content), "etag": etag},
            tmp_path,
            "train",
            resume_partial=True,
        )


def test_fetcher_integrity_receipt_hashes_media_and_reports_template_overlap(tmp_path):
    import json

    import scripts.fetch_clevrer_probe as fetch_probe

    rows = {"train": [], "validation": []}
    for split, scene in (("train", 1), ("validation", 10_001)):
        media = f"\x00\x00\x00\x18ftyp{split}".encode()
        relative = f"media/{split}/video.mp4"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(media)
        rows[split].append(
            {
                "id": f"clevrer-{split}-{scene}-q0",
                "split": split,
                "source": f"MIT-IBM/CLEVRER@{fetch_probe.REVISION}",
                "scene_group_id": f"clevrer:{split}:{scene}",
                "media_path": relative,
                "media_sha256": fetch_probe.sha256(media),
                "question": "Which object is blue?",
                "options": ["yes", "no"],
            }
        )
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "validation.jsonl").write_text("{}\n", encoding="utf-8")

    gate_path = fetch_probe._write_integrity_receipt(
        tmp_path,
        rows["train"],
        rows["validation"],
        allow_question_template_overlap=True,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    assert gate["status"] == "passed"
    assert gate["all_media_sha256_verified"] is True
    assert gate["group_integrity"]["train_validation_scene_overlap"] == 0
    assert gate["group_integrity"]["train_validation_media_overlap"] == 0
    assert gate["normalized_question_option_overlap"] == 1


def test_archive_worker_results_merge_scene_maps_and_io_counters():
    import scripts.fetch_clevrer_probe as fetch_probe

    merged = fetch_probe._merge_archive_worker_results(
        [({1: ("a.mp4", 10, "a" * 64)}, 100, 3, 1), ({2: ("b.mp4", 20, "b" * 64)}, 200, 4, 2)]
    )

    assert merged == {
        "media_by_scene": {1: ("a.mp4", 10, "a" * 64), 2: ("b.mp4", 20, "b" * 64)},
        "range_bytes_read": 300,
        "range_request_count": 7,
        "reused_media_count": 3,
    }


def test_video_probe_selects_eight_deterministic_temporal_positions():
    indices = _uniform_indices(64, requested=8)

    assert indices == [0, 9, 18, 27, 36, 45, 54, 63]


def test_video_features_share_one_forward_per_underlying_video(tmp_path, monkeypatch):
    import scripts.train_frozen_video_probe as video_probe

    class Encoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, clip):
            self.calls += 1
            return torch.arange(768, dtype=torch.float32).reshape(1, 1, 768).expand(
                clip.shape[0], 2, 768
            )

    rows = [
        {"id": "question-a", "media_sha256": "same-video", "media_path": "scene.mp4"},
        {"id": "question-b", "media_sha256": "same-video", "media_path": "scene.mp4"},
    ]
    monkeypatch.setattr(
        video_probe, "_decode_video", lambda _path: np.zeros((8, 2, 2, 3), dtype=np.uint8)
    )
    def transform(_frames):
        return (torch.zeros((3, 8, 2, 2)),)

    encoder = Encoder()

    features, _, unique_videos, cache = _video_features(
        rows, tmp_path, encoder, transform, torch.device("cpu")
    )

    assert encoder.calls == 1
    assert unique_videos == 1
    assert list(cache) == ["same-video"]
    assert features.shape == (2, 768)
    assert torch.equal(features[0], features[1])
