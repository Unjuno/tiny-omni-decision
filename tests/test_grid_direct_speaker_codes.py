import io
import json
import tarfile
import zipfile

import pytest

from scripts.audit_grid_direct_speaker_codes import audit_speaker_bundle


def _tar(path, directory, suffix, codes):
    with tarfile.open(path, 'w') as archive:
        for code in codes:
            payload = code.encode()
            info = tarfile.TarInfo(f'{directory}/{code}{suffix}')
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _zip_tail(path, codes):
    with zipfile.ZipFile(path, 'w') as archive:
        for code in codes:
            archive.writestr(f's13/{code}.mpg', b'not read by the audit')
        archive.writestr('s13/._bbae1s.mpg', b'sidecar')
    data = path.read_bytes()
    return data, 0


def test_audit_reports_matched_source_and_frozen_codes(tmp_path):
    codes = {'bbae1s', 'bbae2p'}
    audio = tmp_path / 'audio.tar'
    alignment = tmp_path / 'alignment.tar'
    video = tmp_path / 'video.zip'
    frozen = tmp_path / 'validation.json'
    _tar(audio, 's13', '.wav', codes)
    _tar(alignment, 'align', '.align', codes)
    _zip_tail(video, codes)
    frozen.write_text(json.dumps([{'speaker': 's13', 'code': c} for c in codes]))

    # Full ZIP bytes are accepted as a tail beginning at offset zero.
    result = audit_speaker_bundle(
        audio_tar=audio,
        alignment_tar=alignment,
        video_tail=video,
        video_range_start=0,
        video_total_size=video.stat().st_size,
        frozen_validation=frozen,
        speaker='s13',
    )

    assert result['status'] == 'PASS_DIRECT_SOURCE_BUNDLE_AND_FROZEN_SPLIT'
    assert result['source_filename_overlaps'] == {
        'audio_alignment': 2,
        'audio_video': 2,
        'alignment_video': 2,
    }
    assert result['source_counts']['video_mpg_codes'] == 2
    assert result['media_payload_read'] is False


def test_audit_distinguishes_frozen_split_mismatch(tmp_path):
    codes = {'bbae1s', 'bbae2p'}
    audio = tmp_path / 'audio.tar'
    alignment = tmp_path / 'alignment.tar'
    video = tmp_path / 'video.zip'
    frozen = tmp_path / 'validation.json'
    _tar(audio, 's13', '.wav', codes)
    _tar(alignment, 'align', '.align', codes)
    _zip_tail(video, codes)
    frozen.write_text(json.dumps([{'speaker': 's13', 'code': 'bbab8n'}]))

    result = audit_speaker_bundle(
        audio_tar=audio,
        alignment_tar=alignment,
        video_tail=video,
        video_range_start=0,
        video_total_size=video.stat().st_size,
        frozen_validation=frozen,
        speaker='s13',
    )

    assert result['status'] == 'OFFICIAL_BUNDLE_MATCH_FROZEN_SPLIT_MISMATCH'
    assert result['frozen_validation_overlap'] == {
        'audio': 0,
        'alignment': 0,
        'video': 0,
    }


def test_audit_rejects_unavailable_or_truncated_zip_tail(tmp_path):
    codes = {'bbae1s'}
    audio = tmp_path / 'audio.tar'
    alignment = tmp_path / 'alignment.tar'
    video = tmp_path / 'video.zip'
    frozen = tmp_path / 'validation.json'
    _tar(audio, 's13', '.wav', codes)
    _tar(alignment, 'align', '.align', codes)
    _zip_tail(video, codes)
    frozen.write_text(json.dumps([{'speaker': 's13', 'code': 'bbae1s'}]))

    with pytest.raises(ValueError, match='through end of file'):
        audit_speaker_bundle(
            audio_tar=audio,
            alignment_tar=alignment,
            video_tail=video,
            video_range_start=1,
            video_total_size=video.stat().st_size,
            frozen_validation=frozen,
            speaker='s13',
        )
