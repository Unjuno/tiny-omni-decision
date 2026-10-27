from scripts.select_openimages_train_candidates import select_candidates


def test_selection_excludes_validation_and_deduplicates_author():
    eligible = {str(i): {"positive": {"a"}, "negative": {"b", "c", "d"}}
                for i in range(4)}
    metadata = {
        str(i): {"OriginalMD5": str(i),
                 "AuthorProfileURL": f"https://www.flickr.com/people/{i // 2}/"}
        for i in range(4)
    }
    selected, counts = select_candidates(eligible, metadata, set(), {"0"}, {"3"}, set(), ["a"])
    assert {row["image_id"] for row in selected} == {"1", "2"}
    assert counts == {"a": 2}


def test_selection_is_deterministic_and_disjoint_from_prior_train():
    eligible = {"one": {"positive": {"a", "b"}, "negative": {"c", "d", "e"}}}
    metadata = {"one": {"OriginalMD5": "x",
                        "AuthorProfileURL": "https://www.flickr.com/people/user/"}}
    args = (eligible, metadata, set(), set(), set(), set(), ["a", "b"])
    assert select_candidates(*args) == select_candidates(*args)
    assert len(select_candidates(*args)[0]) == 1
    assert select_candidates(eligible, metadata, {"one"}, set(), set(), set(), ["a"])[0] == []
