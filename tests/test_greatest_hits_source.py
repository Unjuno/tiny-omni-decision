import io
import zipfile

import pytest


def member_fixture(content=b"one two three\n", compression=zipfile.ZIP_DEFLATED):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("vis-data-256/train.txt", content)
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        item = archive.infolist()[0]
        entry = dict(name=item.filename, offset=item.header_offset, compression=item.compress_type,
                     bytes=item.file_size, compressed=item.compress_size, crc32=f"{item.CRC:08x}")
    data = stream.getvalue()
    return entry, lambda start, size: data[start:start + size]


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_exact_member_extraction(compression):
    from scripts.greatest_hits_source import member_bytes

    entry, read = member_fixture(compression=compression)
    assert member_bytes(entry, read) == b"one two three\n"


@pytest.mark.parametrize("field,value", [
    ("bytes", 1), ("crc32", "00000000"), ("name", "vis-data-256/other.txt"),
    ("name", "../../outside.txt"), ("compression", 12), ("bytes", 40 * 2**20),
])
def test_corrupt_or_unbounded_member_rejected(field, value):
    from scripts.greatest_hits_source import member_bytes

    entry, read = member_fixture(b"x" * 1000)
    entry[field] = value
    with pytest.raises(ValueError):
        member_bytes(entry, read)


def test_exact_http_range_and_version_required():
    from scripts.greatest_hits_source import validate_range

    good = 'HTTP/1.1 206 Partial Content\r\nETag: "fixed"\r\nContent-Range: bytes 2-4/10\r\n'
    validate_range(good, b"abc", 2, 3, 10, '"fixed"')
    for header, body in [(good.replace("206", "200"), b"abc"),
                         (good.replace("fixed", "changed"), b"abc"),
                         (good.replace("2-4/10", "2-5/10"), b"abc"), (good, b"ab")]:
        with pytest.raises(ValueError):
            validate_range(header, body, 2, 3, 10, '"fixed"')


def test_train_selection_is_order_invariant_and_rejects_test_overlap():
    from scripts.greatest_hits_source import parse_ids, select_train

    train = [f"2015-02-16-16-58-{i:02d}" for i in range(10)]
    test = ["2015-02-17-16-58-01"]
    assert parse_ids(("\n".join(train) + "\n").encode()) == train
    a = select_train(train, test, set(train + test))
    assert len(a) == 3 and set(a) <= set(train)
    assert a == select_train(train[::-1], test, set(train + test))
    for tr, te, parents in [(train + train[:1], test, set(train + test)),
                            (train, test + train[:1], set(train + test)),
                            (train, test, set(train))]:
        with pytest.raises(ValueError):
            select_train(tr, te, parents)
    with pytest.raises(ValueError):
        parse_ids(b"../../escape\n")


def test_times_preserves_unknown_labels_and_partial_annotations():
    from scripts.greatest_hits_source import parse_times

    rows = parse_times(b"1.2 leaf scratch scatter\n2.3 None hit None\n")
    assert rows == [dict(seconds=1.2, fields=["leaf", "scratch", "scatter"]),
                    dict(seconds=2.3, fields=[None, "hit", None])]
    # Do not reject or map an as-yet-unreviewed category into a negative class.
    assert parse_times(b"0.1 unreviewed unreviewed unreviewed\n")[0]["fields"][0] == "unreviewed"


@pytest.mark.parametrize("payload", [b"", b"nan leaf hit static\n", b"-1 leaf hit static\n",
                                    b"1 leaf hit\n", b"2 leaf hit static\n1 leaf hit static\n",
                                    b"1 leaf hit static\n1 leaf hit static\n"])
def test_times_rejects_bad_schema_and_timestamps(payload):
    from scripts.greatest_hits_source import parse_times

    with pytest.raises(ValueError):
        parse_times(payload)
