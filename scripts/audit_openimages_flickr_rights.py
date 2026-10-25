"""Check selected Open Images Flickr landing pages without fetching media."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

EXPECTED_LICENSE = "https://creativecommons.org/licenses/by/2.0/"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (compatible; OpenImagesRightsAudit/1.0)"
MAX_PAGE_BYTES = 2_000_000


class _FlickrRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlparse(urljoin(req.full_url, newurl))
        if target.scheme != "https" or target.hostname not in {"www.flickr.com", "flickr.com"}:
            raise ValueError("Flickr landing page redirected outside the HTTPS Flickr host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _JsonLdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._collecting = False
        self._chunks: list[str] = []
        self.documents: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        attributes = {key.lower(): value for key, value in attrs}
        if (attributes.get("type") or "").lower() == "application/ld+json":
            self._collecting = True
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._collecting:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._collecting:
            self.documents.append("".join(self._chunks))
            self._collecting = False
            self._chunks = []


def _walk_image_objects(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _walk_image_objects(item)
    elif isinstance(value, dict):
        kind = value.get("@type", [])
        kinds = {kind} if isinstance(kind, str) else set(kind or [])
        if "ImageObject" in kinds:
            yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _walk_image_objects(child)


def _normalized_name(value: str | None) -> str:
    return "".join(char for char in (value or "").casefold() if char.isalnum())


def _normalized_page_url(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in {"www.flickr.com", "flickr.com"}:
        return None
    return parsed.hostname.removeprefix("www."), parsed.path.rstrip("/")


def _flickr_profile_identity(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in {"www.flickr.com", "flickr.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"people", "photos"}:
        return parsed.hostname.removeprefix("www."), parts[1].casefold()
    return None


def parse_flickr_image_page(
    html: str,
    *,
    expected_landing_url: str,
    expected_author: str,
    expected_author_profile_url: str | None = None,
) -> dict[str, Any]:
    collector = _JsonLdCollector()
    collector.feed(html)
    objects: list[dict[str, Any]] = []
    for document in collector.documents:
        try:
            payload = json.loads(document)
        except json.JSONDecodeError:
            continue
        objects.extend(_walk_image_objects(payload))
    expected_page = _normalized_page_url(expected_landing_url)
    matching = [
        item
        for item in objects
        if _normalized_page_url(item.get("acquireLicensePage")) == expected_page
    ]
    if len(matching) != 1:
        return {
            "status": "image_object_missing_or_ambiguous",
            "license_url": None,
            "author_name": None,
            "author_profile_url": None,
            "page_matches": False,
            "author_name_matches": False,
            "author_profile_matches": False,
            "author_matches": False,
        }
    item = matching[0]
    author = item.get("author")
    if isinstance(author, list):
        author = author[0] if len(author) == 1 else {}
    if not isinstance(author, dict):
        author = {}
    author_name = author.get("name")
    profile_url = author.get("url")
    license_url = item.get("license")
    page_matches = _normalized_page_url(item.get("acquireLicensePage")) == expected_page
    author_name_matches = _normalized_name(author_name) == _normalized_name(expected_author)
    expected_profile = _flickr_profile_identity(expected_author_profile_url)
    author_profile_matches = expected_profile is not None and (
        _flickr_profile_identity(profile_url) == expected_profile
    )
    author_matches = author_name_matches or author_profile_matches
    if not page_matches:
        status = "landing_page_mismatch"
    elif license_url != EXPECTED_LICENSE:
        status = "license_mismatch"
    elif not author_matches:
        status = "author_mismatch"
    else:
        status = "metadata_confirmed_on_landing_page"
    return {
        "status": status,
        "license_url": license_url,
        "author_name": author_name,
        "author_profile_url": profile_url,
        "page_matches": page_matches,
        "author_name_matches": author_name_matches,
        "author_profile_matches": author_profile_matches,
        "author_matches": author_matches,
    }


def check_one_receipt(row: dict[str, Any], *, timeout: float = 20.0) -> dict[str, Any]:
    landing_url = row.get("OriginalLandingURL")
    parsed = urlparse(landing_url or "")
    if parsed.scheme != "https" or parsed.hostname != "www.flickr.com":
        result = {
            "status": "rejected_landing_url_host",
            "http_status": None,
            "final_landing_url": None,
            "page_metadata": None,
        }
    else:
        request = Request(landing_url, headers={"User-Agent": USER_AGENT})
        try:
            opener = build_opener(_FlickrRedirectHandler)
            with opener.open(request, timeout=timeout) as response:
                final_url = response.geturl()
                final = urlparse(final_url)
                body = response.read(MAX_PAGE_BYTES + 1)
                if final.scheme != "https" or final.hostname not in {
                    "www.flickr.com",
                    "flickr.com",
                }:
                    result = {
                        "status": "rejected_redirect_host",
                        "http_status": response.status,
                        "final_landing_url": None,
                        "page_metadata": None,
                    }
                elif len(body) > MAX_PAGE_BYTES:
                    result = {
                        "status": "page_too_large",
                        "http_status": response.status,
                        "final_landing_url": final_url,
                        "page_metadata": None,
                    }
                else:
                    page = parse_flickr_image_page(
                        body.decode(response.headers.get_content_charset() or "utf-8", "replace"),
                        expected_landing_url=landing_url,
                        expected_author=row.get("Author", ""),
                        expected_author_profile_url=row.get("AuthorProfileURL"),
                    )
                    result = {
                        "status": page["status"],
                        "http_status": response.status,
                        "final_landing_url": final_url,
                        "page_metadata": page,
                    }
        except HTTPError as exc:
            result = {
                "status": f"http_error_{exc.code}",
                "http_status": exc.code,
                "final_landing_url": None,
                "page_metadata": None,
            }
        except ValueError as exc:
            status = (
                "rejected_redirect_host"
                if "redirected outside" in str(exc)
                else f"request_error_{type(exc).__name__}"
            )
            result = {
                "status": status,
                "http_status": None,
                "final_landing_url": None,
                "page_metadata": None,
            }
        except (TimeoutError, URLError, OSError) as exc:
            result = {
                "status": f"request_error_{type(exc).__name__}",
                "http_status": None,
                "final_landing_url": None,
                "page_metadata": None,
            }
    return row | {
        "landing_page_checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "landing_page_check_status": result["status"],
        "landing_page_http_status": result["http_status"],
        "landing_page_final_url": result["final_landing_url"],
        "landing_page_metadata": result["page_metadata"],
        "page_license_and_attribution_match": (
            result["status"] == "metadata_confirmed_on_landing_page"
        ),
        "individual_rights_verified": False,
    }


def reconcile_saved_page_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Reapply the current identity rules to a previously saved page observation."""
    metadata = row.get("landing_page_metadata")
    if not isinstance(metadata, dict):
        return row
    name_matches = _normalized_name(metadata.get("author_name")) == _normalized_name(
        row.get("Author")
    )
    expected_profile = _flickr_profile_identity(row.get("AuthorProfileURL"))
    page_profile_matches = expected_profile is not None and (
        _flickr_profile_identity(metadata.get("author_profile_url")) == expected_profile
    )
    author_matches = name_matches or page_profile_matches
    if not metadata.get("page_matches"):
        status = "landing_page_mismatch"
    elif metadata.get("license_url") != EXPECTED_LICENSE:
        status = "license_mismatch"
    elif not author_matches:
        status = "author_mismatch"
    else:
        status = "metadata_confirmed_on_landing_page"
    return row | {
        "landing_page_check_status": status,
        "landing_page_metadata": metadata
        | {
            "author_name_matches": name_matches,
            "author_profile_matches": page_profile_matches,
            "author_matches": author_matches,
        },
        "page_license_and_attribution_match": status == "metadata_confirmed_on_landing_page",
        "individual_rights_verified": False,
    }


def check_receipts(rows: list[dict[str, Any]], *, workers: int = 3) -> list[dict[str, Any]]:
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return [
            reconcile_saved_page_metadata(row)
            for row in executor.map(check_one_receipt, rows)
        ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("receipt JSONL must contain objects")
    return rows


def summarize_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("landing_page_check_status", "missing_status")
        counts[status] = counts.get(status, 0) + 1
    return {
        "rows": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "page_declarations_confirmed_count": sum(
            row.get("page_license_and_attribution_match") is True for row in rows
        ),
        "all_page_declarations_confirmed": bool(rows)
        and all(row.get("page_license_and_attribution_match") is True for row in rows),
        "independent_ownership_verification_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipts", type=Path, required=True)
    parser.add_argument("--output-receipts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    destinations = (args.output_receipts.resolve(), args.summary.resolve())
    if any(path.is_relative_to(REPOSITORY_ROOT) for path in destinations):
        raise ValueError(
            "rights check outputs containing source attribution must be outside the repo"
        )
    if destinations[0] == destinations[1] or any(path.exists() for path in destinations):
        raise FileExistsError("refusing to overwrite or alias rights check outputs")
    checked = check_receipts(load_jsonl(args.input_receipts), workers=args.workers)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in checked)
    args.output_receipts.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipts.write_text(payload, encoding="utf-8")
    summary = summarize_checks(checked) | {
        "checked_receipts_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "raw_media_downloaded": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
