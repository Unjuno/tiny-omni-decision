import copy
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip('torch')

from scripts.lombard_readout_data import (  # noqa: E402
    CHECKPOINT_SHA,
    SOURCE_REVISION,
    read_bank,
    validate_bank,
)


def fixture(kind):
    rows = [{'source_id': str(i), 'split': 'train'} for i in range(4)]
    media = {r['source_id']: dict(audio_sha256='a'*64, video_sha256='b'*64,
                                samples=32000, video_frames=16,
                                first_video_pts=0., last_video_pts=1.) for r in rows}
    info = dict(status='PASS', dtype='float32', training=False,
                rows=[dict(source_id=r['source_id'], split=r['split'],
                           input_sha256=('a' if kind == 'audio' else 'b')*64,
                           offset=i*100, tokens=100, samples=32000, repeat_bitwise_equal=True,
                           selected_indices=list(range(16)),
                           selected_pts=np.linspace(0, 1, 16).tolist())
                      for i, r in enumerate(rows)],
                model_parameters=8208384, all_parameters_unchanged=True,
                all_inputs_complete=True, parameters=114365184,
                parameters_unchanged=True, all_features_reload_verified=True,
                all_media_hashes_verified=True, all_expected_ids_present=True,
                parameter_sha256_before='c'*64, parameter_sha256_after='c'*64,
                native_reference_checks=4,
                config=dict(frames=16, height=288, width=288, dtype='float32',
                            source_revision=SOURCE_REVISION, checkpoint_sha256=CHECKPOINT_SHA,
                            training=False, labels_transcripts_alignments_as_input=False,
                            total_rows=4))
    if kind == 'video':
        del info['dtype']  # actual native report declares dtype inside config
    return info, np.zeros((4, 384 if kind == 'audio' else 768), dtype='float32'), rows, media


@pytest.mark.parametrize('kind', ['audio', 'video'])
def test_complete_native_bank_fixture_passes(kind):
    info, value, rows, media = fixture(kind)
    validate_bank(info, value, rows, kind, media)


@pytest.mark.parametrize('fault', ['running', 'shape', 'dtype', 'nan', 'order', 'duplicate',
                                 'input_sha', 'split', 'frozen', 'frames', 'source', 'reload',
                                 'weights_hash', 'time_order', 'sampling'])
def test_partial_or_wrong_video_bank_fails_before_training(fault):
    info, value, rows, media = fixture('video')
    if fault == 'running':
        info['status'] = 'RUNNING'  # fully allocated zeros are not a completed cache
    elif fault == 'shape':
        value = value[:, :767]
    elif fault == 'dtype':
        value = value.astype('float16')
    elif fault == 'nan':
        value[0, 0] = np.nan
    elif fault == 'order':
        info['rows'].reverse()
    elif fault == 'duplicate':
        info['rows'][1] = copy.deepcopy(info['rows'][0])
    elif fault == 'input_sha':
        info['rows'][0]['input_sha256'] = 'd'*64
    elif fault == 'split':
        info['rows'][0]['split'] = 'validation'
    elif fault == 'frozen':
        info['parameters_unchanged'] = False
    elif fault == 'frames':
        info['config']['frames'] = 8
    elif fault == 'source':
        info['config']['source_revision'] = 'd'*40
    elif fault == 'reload':
        info['all_features_reload_verified'] = False
    elif fault == 'weights_hash':
        info['parameter_sha256_after'] = 'd'*64
    elif fault == 'time_order':
        info['rows'][0]['selected_pts'].reverse()
    else:
        info['rows'][0]['selected_indices'][0] = 1
    with pytest.raises(ValueError):
        validate_bank(info, value, rows, 'video', media)


def test_file_hash_binding_cannot_be_bypassed_by_plausible_metadata(tmp_path):
    info, value, rows, media = fixture('audio')
    np.save(tmp_path/'pooled.npy', value)
    info['feature_hashes'] = {'pooled.npy': hashlib.sha256(
        (tmp_path/'pooled.npy').read_bytes()).hexdigest()}
    (tmp_path/'result.json').write_text(json.dumps(info))
    assert read_bank(tmp_path, rows, 'audio', media)[0].shape == (4, 384)
    info['feature_hashes']['pooled.npy'] = 'f'*64
    (tmp_path/'result.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='hash'):
        read_bank(tmp_path, rows, 'audio', media)
