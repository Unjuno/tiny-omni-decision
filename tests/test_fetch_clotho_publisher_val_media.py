import pytest

from scripts.fetch_clotho_publisher_val_media import unique_media_rows


def media_row(source_id="123", question="Question one?", file_name="Clip.wav", uploader="person"):
    return {
        "source_id": source_id,
        "normalized_question": question.casefold(),
        "file_name": file_name,
        "uploader": uploader,
        "license": "http://creativecommons.org/publicdomain/zero/1.0/",
    }


def test_repeated_questions_share_one_acquired_audio_asset():
    rows = [media_row(question="Question one?"), media_row(question="Question two?")]

    assert unique_media_rows(rows) == [rows[0]]


def test_audio_asset_identity_conflict_fails_closed():
    rows = [media_row(), media_row(file_name="Different.wav")]

    with pytest.raises(ValueError, match="inconsistent identity metadata"):
        unique_media_rows(rows)
