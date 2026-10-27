import pytest

from scripts.audit_grid_candidate_coverage import coverage


def test_excludes_prior_and_reports_missing_video_group():
    names = ['alignments/s1/bbaf2n.align', 'alignments/s21/bbaf3s.align']
    result = coverage(names, {'bbaf2n'})
    assert result['overlap_codes'] == 1
    assert result['eligible_alignment_rows'] == result['s21_audio_only_eligible'] == 1


def test_duplicate_code_fails():
    with pytest.raises(ValueError, match='not unique'):
        coverage(['alignments/s1/bbaf2n.align', 'alignments/s2/bbaf2n.align'], set())
