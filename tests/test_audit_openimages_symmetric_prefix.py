import pytest

pytest.importorskip("torch")
import scripts.audit_openimages_symmetric_prefix as audit  # noqa: E402


def test_count_only_explicit_negatives_and_drop_final_group():
    scope = {f"c{i}" for i in range(11)}
    groups = [
        {"positive_labels": {"c0"}, "negative_labels": {"c1", "c2", "c3"}},
        {"positive_labels": {"c0"}, "negative_labels": {"c1"}},
        {"positive_labels": {"c0"}, "negative_labels": {"c1", "c2", "c3"}},
    ]
    n, counts = audit.count_groups(groups, scope)
    assert n == 2 and counts["c0"]["positive_records"] == 2
    assert counts["c0"]["four_choice_eligible"] == 1
    assert counts["c0"]["negative_count_1"] == 1


def test_range_contract_rejects_full_body_and_mismatch(monkeypatch):
    monkeypatch.setattr(audit, "SIZE", 4)
    monkeypatch.setattr(audit, "TOTAL", 9)

    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-3/9", "ETag": f'"{audit.ETAG}"'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            return iter([b"abcd"])

    class Session:
        def get(self, url, headers, stream, timeout):
            return Response()

    assert audit.fetch_prefix(Session()) == b"abcd"
    monkeypatch.setattr(Response, "status_code", 200)
    with pytest.raises(ValueError, match="range"):
        audit.fetch_prefix(Session())
