import json
from urllib.request import Request

import pytest

from scripts.audit_openimages_flickr_rights import (
    _FlickrRedirectHandler,
    parse_flickr_image_page,
    reconcile_saved_page_metadata,
    summarize_checks,
)


def _html(*, license_url, author="Example Author", page="123"):
    payload = {
        "@context": "https://schema.org/",
        "@type": "ImageObject",
        "license": license_url,
        "acquireLicensePage": f"https://www.flickr.com/photos/example/{page}",
        "author": {
            "@type": "Person",
            "name": author,
            "url": "https://www.flickr.com/photos/example",
        },
    }
    return (
        "<html><script type='application/ld+json'>"
        + json.dumps(payload)
        + "</script></html>"
    )


def test_flickr_rights_parser_confirms_matching_page_license_and_author():
    result = parse_flickr_image_page(
        _html(license_url="https://creativecommons.org/licenses/by/2.0/"),
        expected_landing_url="https://www.flickr.com/photos/example/123/",
        expected_author="example author",
    )

    assert result["status"] == "metadata_confirmed_on_landing_page"
    assert result["page_matches"] is True
    assert result["author_matches"] is True


def test_flickr_rights_parser_does_not_accept_changed_license_or_author():
    changed_license = parse_flickr_image_page(
        _html(license_url="https://www.flickrhelp.com/terms"),
        expected_landing_url="https://www.flickr.com/photos/example/123",
        expected_author="Example Author",
    )
    wrong_author = parse_flickr_image_page(
        _html(license_url="https://creativecommons.org/licenses/by/2.0/", author="Someone Else"),
        expected_landing_url="https://www.flickr.com/photos/example/123",
        expected_author="Example Author",
    )

    assert changed_license["status"] == "license_mismatch"
    assert wrong_author["status"] == "author_mismatch"


def test_flickr_rights_parser_accepts_same_author_profile_with_changed_display_name():
    payload = {
        "@context": "https://schema.org/",
        "@type": "ImageObject",
        "license": "https://creativecommons.org/licenses/by/2.0/",
        "acquireLicensePage": "https://www.flickr.com/photos/example/123",
        "author": {
            "@type": "Person",
            "name": "Updated Display Name",
            "url": "https://www.flickr.com/photos/example",
        },
    }
    result = parse_flickr_image_page(
        "<script type='application/ld+json'>" + json.dumps(payload) + "</script>",
        expected_landing_url="https://www.flickr.com/photos/example/123",
        expected_author="Previous Display Name",
        expected_author_profile_url="https://www.flickr.com/people/example/",
    )

    assert result["status"] == "metadata_confirmed_on_landing_page"
    assert result["author_name_matches"] is False
    assert result["author_profile_matches"] is True


def test_flickr_rights_parser_rejects_missing_or_wrong_page_identity():
    missing = parse_flickr_image_page(
        "<html><script type='application/ld+json'>{}</script></html>",
        expected_landing_url="https://www.flickr.com/photos/example/123",
        expected_author="Example Author",
    )
    mismatched_page = parse_flickr_image_page(
        _html(license_url="https://creativecommons.org/licenses/by/2.0/", page="999"),
        expected_landing_url="https://www.flickr.com/photos/example/123",
        expected_author="Example Author",
    )

    assert missing["status"] == "image_object_missing_or_ambiguous"
    assert mismatched_page["status"] == "image_object_missing_or_ambiguous"


def test_summary_counts_page_declarations_without_claiming_owner_verification():
    report = summarize_checks(
        [
            {
                "landing_page_check_status": "metadata_confirmed_on_landing_page",
                "page_license_and_attribution_match": True,
                "individual_rights_verified": False,
            },
            {
                "landing_page_check_status": "license_mismatch",
                "page_license_and_attribution_match": False,
                "individual_rights_verified": False,
            },
        ]
    )

    assert report["rows"] == 2
    assert report["page_declarations_confirmed_count"] == 1
    assert report["all_page_declarations_confirmed"] is False
    assert report["independent_ownership_verification_performed"] is False


def test_flickr_page_auditor_rejects_redirects_to_other_hosts():
    request = Request("https://www.flickr.com/photos/example/123")

    with pytest.raises(ValueError, match="redirected outside"):
        _FlickrRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://attacker.example/image.jpg",
        )


def test_saved_page_metadata_can_verify_same_profile_after_display_name_change():
    row = {
        "Author": "Previous Display Name",
        "AuthorProfileURL": "https://www.flickr.com/people/example/",
        "landing_page_check_status": "author_mismatch",
        "individual_rights_verified": False,
        "landing_page_metadata": {
            "page_matches": True,
            "license_url": "https://creativecommons.org/licenses/by/2.0/",
            "author_name": "Updated Display Name",
            "author_profile_url": "https://www.flickr.com/photos/example",
        },
    }

    result = reconcile_saved_page_metadata(row)

    assert result["landing_page_check_status"] == "metadata_confirmed_on_landing_page"
    assert result["individual_rights_verified"] is False
    assert result["page_license_and_attribution_match"] is True
    assert result["landing_page_metadata"]["author_name_matches"] is False
    assert result["landing_page_metadata"]["author_profile_matches"] is True
