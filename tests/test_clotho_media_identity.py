import pytest

from scripts.clotho_media_identity import media_basename


def test_reviewed_alias_only():
    assert media_basename(dict(file_name="souffle_me_tallique.wav",
                               source_id="16854", uploader="deegnot")) == "souffle_me.tallique.wav"
    assert media_basename(dict(file_name="other_name.wav")) == "other_name.wav"


@pytest.mark.parametrize("source,uploader", [("16855", "deegnot"), ("16854", "other")])
def test_wrong_provenance_rejected(source, uploader):
    with pytest.raises(ValueError):
        media_basename(dict(file_name="souffle_me_tallique.wav",
                            source_id=source, uploader=uploader))
