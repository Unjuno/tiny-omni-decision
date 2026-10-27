import io
import json
import tarfile
import zipfile

import pytest

from scripts.smoke_grid_direct_speaker_quartet import (
    read_audio_members,
    read_video_members,
    validate_json,
)


def test_validate_json_requires_normalized_probabilities_and_exact_ids():
    ids = ['a', 'b', 'c', 'd']
    valid = json.dumps({
        'selected_option_id': 'a',
        'probabilities': dict(zip(ids, [0.4, 0.3, 0.2, 0.1], strict=True)),
    })
    assert validate_json(valid, ids)['selected_option_id'] == 'a'

    invalid = json.dumps({'selected_option_id': 'a', 'probabilities': {'a': 1.0}})
    with pytest.raises(ValueError, match='invalid output probabilities'):
        validate_json(invalid, ids)


def test_direct_member_readers_retrieve_only_the_fixed_speaker_codes(tmp_path):
    audio_path = tmp_path / 's30.tar'
    with tarfile.open(audio_path, 'w') as archive:
        for name in ('s30/aaaaaa.wav', 's30/bbbbbb.wav', 's29/cccccc.wav'):
            data = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    video_path = tmp_path / 's30.zip'
    with zipfile.ZipFile(video_path, 'w') as archive:
        archive.writestr('s30/aaaaaa.mpg', b'first-video')
        archive.writestr('s30/bbbbbb.mpg', b'second-video')

    audio = read_audio_members(audio_path, 's30', ['aaaaaa', 'bbbbbb'])
    video = read_video_members(video_path, 's30', ['aaaaaa', 'bbbbbb'])
    assert audio == {
        'aaaaaa': b's30/aaaaaa.wav',
        'bbbbbb': b's30/bbbbbb.wav',
    }
    assert video == {'aaaaaa': b'first-video', 'bbbbbb': b'second-video'}


def test_direct_audio_reader_fails_closed_for_missing_assets(tmp_path):
    audio_path = tmp_path / 's30.tar'
    with tarfile.open(audio_path, 'w') as archive:
        data = b'only-one'
        info = tarfile.TarInfo('s30/aaaaaa.wav')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError, match='missing from direct speaker archive'):
        read_audio_members(audio_path, 's30', ['aaaaaa', 'bbbbbb'])
