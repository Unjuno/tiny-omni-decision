from copy import deepcopy

import pytest


def module():
    from scripts import native_audio_sources
    return native_audio_sources


def sources():
    from scripts.lombard_grid_corpus import decode_code
    rows, media, events = {}, [], {}
    for index, split in enumerate(('train', 'validation')):
        code = ['bbab2n', 'prwr3a'][index]
        source_id = f's{11+index}_l_{code}'
        rows[split] = [dict(source_id=source_id, speaker=f's{11+index}',
            speaker_group=f's{11+index}', split=split, actual_code=code,
            intended_code=code, label_status='eligible', targets=decode_code(code),
            assets={'audio': dict(name=f'audio/{source_id}.wav', bytes=100)})]
        media.append(dict(source_id=source_id, split=split, audio_status='PASS',
            video_status='PASS', rate=16000, samples=321, audio_sha256=str(index+1)*64,
            decoded_mono_f32_sha256=str(index+3)*64))
        events[split] = [dict(id=f'rec{index}:0-5', split=split, recording=f'rec{index}',
            room=f'room{index}', source_revision='doi:10.5281/zenodo.7880637',
            start_seconds=0, end_seconds=5,
            media={'audio':dict(path=f'file{index}.wav', sha256=str(index+5)*64, bytes=123)})]
    return rows, media, events


def test_plan_has_separate_split_offsets_and_correct_sample_accounting():
    plan = module().audio_plan(*sources())
    assert [r['source'] for r in plan['rows']] == ['lombard', 'starss', 'lombard', 'starss']
    assert [r['speech_tokens'] for r in plan['rows']] == [2, 250, 2, 250]
    assert [r['speech_offset'] for r in plan['rows']] == [0, 2, 0, 2]
    assert [r['event_offset'] for r in plan['rows']] == [0, 1, 0, 1]
    assert plan['shapes'] == {'train-speech.npy':[252,384], 'train-event.npy':[2,384],
                              'validation-speech.npy':[252,384],
                              'validation-event.npy':[2,384]}
    assert plan['tensor_bytes'] == 508 * 384 * 4


@pytest.mark.parametrize('field', ['speaker', 'speaker_group', 'actual_code'])
def test_lombard_cross_split_identity_is_rejected(field):
    from scripts.lombard_grid_corpus import decode_code
    rows, media, events = sources()
    rows['validation'][0][field] = rows['train'][0][field]
    if field == 'actual_code':
        rows['validation'][0]['targets'] = decode_code(rows['validation'][0][field])
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


@pytest.mark.parametrize('field', ['room', 'recording'])
def test_starss_cross_split_identity_is_rejected(field):
    rows, media, events = sources()
    events['validation'][0][field] = events['train'][0][field]
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


def test_same_audio_asset_cannot_cross_splits():
    rows, media, events = sources()
    events['validation'][0]['media']['audio']['sha256'] = media[0]['audio_sha256']
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


def test_decoded_content_overlap_is_rejected():
    rows, media, events = sources()
    media[1]['decoded_mono_f32_sha256'] = media[0]['decoded_mono_f32_sha256']
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


def test_reserved_splits_are_not_accepted():
    rows, media, events = sources()
    rows['calibration'] = []
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


def test_overlapping_starss_windows_are_not_duplicated():
    rows, media, events = sources()
    second = deepcopy(events['train'][0])
    second.update(id='overlap', start_seconds=1, end_seconds=6)
    events['train'].append(second)
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


def test_invalid_media_and_changed_labels_are_rejected():
    rows, media, events = sources()
    media[0]['audio_status'] = 'FAIL'
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)
    rows, media, events = sources()
    rows['train'][0]['targets']['color'] = 'red'
    with pytest.raises(ValueError):
        module().audio_plan(rows, media, events)


def test_multiple_windows_share_one_asset_but_keep_distinct_offsets():
    rows, media, events = sources()
    second = deepcopy(events['train'][0])
    second.update(id='next', start_seconds=5, end_seconds=10)
    events['train'].append(second)
    plan = module().audio_plan(rows, media, events)
    assert [r['speech_offset'] for r in plan['rows'][:3]] == [0, 2, 252]
    assert plan['counts']['train']['starss']['examples'] == 2
    assert plan['counts']['train']['starss']['assets'] == 1


def test_identical_pcm_from_other_split_is_rejected_after_decode():
    owners = {}
    module().bind_pcm(owners, 'train', 16000, 'a'*64)
    module().bind_pcm(owners, 'train', 16000, 'a'*64)
    with pytest.raises(ValueError):
        module().bind_pcm(owners, 'validation', 16000, 'a'*64)


def wav_bytes(channels=1, samples=321, rate=16000):
    import io
    import wave

    import numpy as np
    content = io.BytesIO()
    with wave.open(content, 'wb') as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.full((samples, channels), 256, dtype='<i2').tobytes())
    return content.getvalue()


def audio_fixture(tmp_path, source):
    import hashlib
    import zipfile

    import numpy as np
    pytest.importorskip('soundfile')
    archive = tmp_path / 'audio.zip'
    row = dict(id='example', source=source, split='train',
               samples=321 if source == 'lombard' else 120000,
               rate=16000 if source == 'lombard' else 24000)
    raw = wav_bytes(1 if source == 'lombard' else 4, row['samples'], row['rate'])
    row.update(media_bytes=len(raw), media_sha256=hashlib.sha256(raw).hexdigest())
    with zipfile.ZipFile(archive, 'w') as output:
        if source == 'lombard':
            row['member'] = 'sample.wav'
            output.writestr(row['member'], raw)
    if source == 'starss':
        path = tmp_path / 'recording.wav'
        path.write_bytes(raw)
        row.update(path=str(path), start_seconds=0, end_seconds=5)
    expected = np.full(row['samples'], 256/32768, dtype=np.float32)
    row['pcm_sha256'] = hashlib.sha256(expected.tobytes()).hexdigest()
    return row, archive, expected


@pytest.mark.parametrize('source', ['lombard', 'starss'])
def test_real_decode_matches_registered_pcm(tmp_path, source):
    import numpy as np
    row, archive, expected = audio_fixture(tmp_path, source)
    outputs = list(module().iter_audio([row], archive))
    assert len(outputs) == 1
    assert np.array_equal(outputs[0][1], expected)
    assert outputs[0][2] == row['pcm_sha256']


@pytest.mark.parametrize('source', ['lombard', 'starss'])
@pytest.mark.parametrize('field', ['media_sha256', 'pcm_sha256', 'media_bytes'])
def test_real_decode_rejects_changed_registered_content(tmp_path, source, field):
    row, archive, _ = audio_fixture(tmp_path, source)
    row[field] = 1 if field == 'media_bytes' else '0'*64
    with pytest.raises(ValueError):
        list(module().iter_audio([row], archive))


def test_starss_changes_after_consumption_are_detected(tmp_path):
    from pathlib import Path
    row, archive, _ = audio_fixture(tmp_path, 'starss')
    iterator = module().iter_audio([row], archive)
    next(iterator)
    with Path(row['path']).open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError):
        list(iterator)
