from scripts.audit_grid_original_content import ENTRY


def test_match_only_real_original_alignment_paths():
    assert ENTRY.fullmatch('alignments/s1/bbaf2n.align')[1] == 'bbaf2n'
    assert not ENTRY.fullmatch('__MACOSX/alignments/s1/._bbaf2n.align')
    assert not ENTRY.fullmatch('../alignments/s1/bbaf2n.align')
