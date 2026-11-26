import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_intphys_dev.py"
SPEC = importlib.util.spec_from_file_location("audit_intphys_dev", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_archive(path: Path, *, labels: tuple[bool, ...] = (True, True, False, False)) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for rendition, label in enumerate(labels, start=1):
            prefix = f"dev/O1/01/{rendition}"
            status = json.dumps({"header": {"is_possible": label}}).encode()
            status_info = tarfile.TarInfo(f"{prefix}/status.json")
            status_info.size = len(status)
            archive.addfile(status_info, io.BytesIO(status))
            for modality, basename in (
                ("scene", "scene"),
                ("depth", "depth"),
                ("masks", "masks"),
            ):
                for frame_number in range(1, 3):
                    frame_info = tarfile.TarInfo(
                        f"{prefix}/{modality}/{basename}_{frame_number:03}.png"
                    )
                    frame_info.size = 1
                    archive.addfile(frame_info, io.BytesIO(b"x"))


def test_audit_accepts_complete_paired_scene_group(tmp_path: Path) -> None:
    archive = tmp_path / "dev.tar.gz"
    _make_archive(archive)

    result = MODULE.audit_archive(archive, expected_frames_per_video=2)

    assert result["video_count"] == 4
    assert result["scene_group_count"] == 1
    assert result["blocks"]["O1"] == {
        "impossible": 2,
        "possible": 2,
        "videos": 4,
    }
    assert result["matched_groups_have_two_labels_each"] is True
    assert result["cross_label_scene_frame_overlap"] == {
        "video_pairs": 4,
        "pairs_with_any_shared_frame": 4,
        "same_index_equal_frame_total": 8,
        "same_index_equal_frame_fraction": 1.0,
        "shared_frame_content_total": 8,
    }


def test_audit_rejects_label_imbalance(tmp_path: Path) -> None:
    archive = tmp_path / "dev.tar.gz"
    _make_archive(archive, labels=(True, True, True, False))

    with pytest.raises(ValueError, match="2 possible and 2 impossible"):
        MODULE.audit_archive(archive, expected_frames_per_video=2)


def test_audit_checks_published_archive_hash(tmp_path: Path) -> None:
    archive = tmp_path / "dev.tar.gz"
    _make_archive(archive)

    with pytest.raises(ValueError, match="MD5 mismatch"):
        MODULE.audit_archive(archive, expected_md5="deadbeef")
