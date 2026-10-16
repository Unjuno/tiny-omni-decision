import copy
import hashlib
import json

import pytest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path):
    cache, proof = tmp_path/'cache', tmp_path/'proof'
    cache.mkdir()
    proof.mkdir()
    (cache/'tokens.npy').write_bytes(b'payload whose hash is attested')
    raw = dict(status='INCOMPLETE',phase='full',
        source_commit='61530adc76935e4b5dbfbabf582ef1ec03cd1405',
        error="ValueError('incomplete coverage or encoder state changed')",
        frozen_before={'model':'unchanged'},frozen_after={'model':'unchanged'},
        expected_events=2,completed_events=2,expected_parents=1,completed_parents=1,
        expected_questions=3,completed_questions=3,matched_old_audio_events=2,
        matched_smoke_events=1,optimizer_updates=0,
        artifacts={'tokens.npy':sha(cache/'tokens.npy')})
    write(cache/'result.json',raw)
    artifacts = {str(cache/'tokens.npy'):sha(cache/'tokens.npy'),
                 str(cache/'result.json'):sha(cache/'result.json')}
    write(proof/'artifact-hashes.json',artifacts)
    verification = dict(status='PASS_FULL_AUDIO_CACHE_REVALIDATION',
        source_commit='933e5dc84cbf125b3a2965f4250fdd0de5ad3ef5',
        source_cache=str(cache),producer_status='INCOMPLETE',
        producer_result_sha256=sha(cache/'result.json'),verified_events=2,
        verified_parents=1,verified_questions=3,input_replay_matches=2,
        feature_digest_matches=4,smoke_feature_matches=2,
        corrected_expected_reference_matches=2,encoder_replayed=False,
        optimizer_updates=0,quality_evaluated=False,sealed_audit_loaded=False,
        calibration_loaded=False,artifacts=artifacts,
        artifact_ledger_sha256=sha(proof/'artifact-hashes.json'))
    write(proof/'result.json',verification)
    return cache,proof,raw,verification


def test_explicit_admission_returns_cache_view_and_preserves_failed_producer(tmp_path):
    from scripts.greatest_hits_audio_admission import load_revalidated_audio

    cache,proof,raw,verification = fixture(tmp_path)
    before = (cache/'result.json').read_bytes()
    view = load_revalidated_audio(cache,proof/'result.json',sha(proof/'result.json'))
    assert view['status'] == 'PASS_COMPLETE_AUDIO_QUERY_CACHE'
    assert view['audio_admission']['producer_status'] == 'INCOMPLETE'
    assert view['audio_admission']['producer_result_sha256'] == sha(cache/'result.json')
    assert view['audio_admission']['verification_sha256'] == sha(proof/'result.json')
    assert raw == json.loads(before) and (cache/'result.json').read_bytes() == before
    assert view['artifacts'] == raw['artifacts']
    assert verification['status'] == 'PASS_FULL_AUDIO_CACHE_REVALIDATION'


@pytest.mark.parametrize('case', ['proof_hash','producer_hash','payload','root','status',
                                 'input_count','feature_count','smoke_count','ledger','quality'])
def test_admission_fails_closed_on_bad_proof_or_saved_files(tmp_path,case):
    from scripts.greatest_hits_audio_admission import load_revalidated_audio

    cache,proof,raw,verification = fixture(tmp_path)
    value = copy.deepcopy(verification)
    if case == 'proof_hash':
        digest = '0'*64
    else:
        if case == 'producer_hash':
            value['producer_result_sha256'] = '0'*64
        elif case == 'payload':
            (cache/'tokens.npy').write_bytes(b'changed')
        elif case == 'root':
            value['source_cache'] = str(tmp_path/'different-cache')
        elif case == 'status':
            value['status'] = 'RUNNING'
        elif case == 'input_count':
            value['input_replay_matches'] = 1
        elif case == 'feature_count':
            value['feature_digest_matches'] = 3
        elif case == 'smoke_count':
            value['smoke_feature_matches'] = 0
        elif case == 'ledger':
            del value['artifacts'][str(cache/'tokens.npy')]
        else:
            value['quality_evaluated'] = True
        write(proof/'result.json',value)
        digest = sha(proof/'result.json')
    with pytest.raises(ValueError):
        load_revalidated_audio(cache,proof/'result.json',digest)
    assert json.loads((cache/'result.json').read_text()) == raw
