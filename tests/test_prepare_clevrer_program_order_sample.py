import json

import pytest

torch = pytest.importorskip("torch")  # noqa: F401
pytest.importorskip("transformers")
pytest.importorskip("av")

from scripts.fetch_clevrer_probe import choose  # noqa: E402
from scripts.prepare_clevrer_program_order_sample import (  # noqa: E402
    _read_member_with_retries,
    discover_previously_observed_scenes,
    select_unused_validation,
)


def _question(question_id, text, program):
    return {
        "question_id": question_id,
        "question": text,
        "question_type": "descriptive",
        "question_subtype": "exist",
        "program": program,
        "answer": "yes",
    }


def _raw_scenes(offset, count=6):
    return [
        {
            "scene_index": offset + scene,
            "video_filename": f"video_{offset + scene:05d}.mp4",
            "questions": [
                _question(0, f"Are objects visible in scene {scene}?", ["objects", "exist"]),
                _question(
                    1,
                    f"What happens before collision in scene {scene}?",
                    ["filter_collision", "before"],
                ),
            ],
        }
        for scene in range(count)
    ]


def test_unused_validation_selector_excludes_prior_scenes_and_content(tmp_path):
    artifacts = tmp_path / "clevrer-artifacts"
    artifacts.mkdir()
    old_row = {"id": "clevrer-validation-10000-q000", "source": "MIT-IBM/CLEVRER"}
    (artifacts / "prior.jsonl").write_text(json.dumps(old_row) + "\n", encoding="utf-8")
    training = choose(_raw_scenes(0), "train", per_type=2, seed=17)

    selected, exclusions, _ = select_unused_validation(
        training,
        _raw_scenes(11_000),
        tmp_path,
        per_type=2,
        seed=17,
    )

    scenes = {int(row["scene_index"]) for row in selected}
    assert 10_000 in exclusions
    assert scenes == {row["scene_index"] for row in choose_validation_for_test(training)}
    assert 10_000 not in scenes
    assert len(selected) == 4
    assert all(
        row["question_type"] in {"static_descriptive", "temporal_descriptive"} for row in selected
    )
    assert {row["question_type"] for row in selected} == {
        "static_descriptive",
        "temporal_descriptive",
    }


def choose_validation_for_test(training):
    from scripts.fetch_clevrer_probe import choose_validation

    content = {
        (row["question"].casefold().strip(), tuple(sorted(row["options"]))) for row in training
    }
    return choose_validation(_raw_scenes(11_000), 2, 17, content, {10_000})


def test_scene_inventory_ignores_non_clevrer_lines_and_records_source_hash(tmp_path):
    artifacts = tmp_path / "clevrer-run"
    artifacts.mkdir()
    path = artifacts / "predictions.jsonl"
    path.write_text(
        json.dumps({"id": "clevrer-validation-12345-q002"})
        + "\n"
        + json.dumps({"id": "unrelated-54321"})
        + "\n",
        encoding="utf-8",
    )

    scenes, sources = discover_previously_observed_scenes(tmp_path)

    assert scenes == {12345}
    assert sources == [
        {
            "path": str(path),
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
    ]


def test_archive_member_read_retries_transient_connection_failure():
    from requests.exceptions import ConnectionError

    class Archive:
        calls = 0

        def read(self, _member):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient range failure")
            return b"video"

    archive = Archive()

    assert _read_member_with_retries(archive, "member.mp4") == b"video"
    assert archive.calls == 2
