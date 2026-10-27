from scripts.select_openimages_quartet_candidates import rank_quartets, select_unique


def test_select_unique_excludes_author_md5_and_prior_id():
    metadata = {str(i): {"OriginalMD5": str(i),
                         "AuthorProfileURL": f"https://flickr.com/people/{i // 2}/"}
                for i in range(4)}
    selected = select_unique(metadata, {"0"}, {"3"}, set())
    assert set(selected) == {"1", "2"}
    assert select_unique(metadata, {"0"}, {"3"}, set()) == selected


def test_rank_quartets_prioritizes_scarce_and_respects_limit():
    groups = [{"rows": [{"image_id": str(i), "target": str(i)} for i in range(4)]},
              {"rows": [{"image_id": str(i + 4), "target": str(i)} for i in range(4)]},
              {"rows": [{"image_id": str(i + 8), "target": str(i + 4)} for i in range(4)]}]
    ranked = rank_quartets(groups, limit=1)
    assert ranked == [groups[2]]
