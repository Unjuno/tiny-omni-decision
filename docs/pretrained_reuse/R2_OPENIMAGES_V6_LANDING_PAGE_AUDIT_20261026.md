# Open Images V6 selected Flickr landing-page audit (2026-10-26)

## Scope and method

Checked the 440 selected train/validation records from the fixed seed-17 pilot
against their Flickr landing pages. The checker fetched landing-page HTML only,
parsed the page's Schema.org `ImageObject`, and matched the page identity,
declared license, and uploader profile with the pinned Open Images metadata.
It did not fetch any photo bytes. The 76 HTTP 403/404 cases were retried once
with two workers; no retry changed a result. Full private per-image receipts,
including the source URLs and attribution fields, remain outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2\openimages-v6-support-audit-20261026\`.

The status `metadata_confirmed_on_landing_page` means the current Flickr page
declares CC BY 2.0, its photo page matches the selected landing page, and either
the page author name or the profile identity matches the source metadata. It
does **not** independently authenticate copyright ownership. The receipt keeps
`individual_rights_verified=false` for every row.

## Results

| Split | Selected | Page declaration and attribution match | 404 | 403 | License mismatch | Page mismatch |
|---|---:|---:|---:|---:|---:|---:|
| Train | 220 | 176 | 33 | 4 | 7 | 0 |
| Validation | 220 | 171 | 34 | 5 | 8 | 2 |
| Total | 440 | 347 | 67 | 9 | 15 | 2 |

Thirty records had a changed display name but the same Flickr profile identity;
these were counted as matching attribution identity while retaining both page
values in the private receipt. A page whose license no longer declares CC BY
2.0, an inaccessible page, or a mismatched landing page is not included among
the 347 page-declaration matches. The retries found no new confirmations.

The [official Open Images V6 documentation](https://storage.googleapis.com/openimages/web/factsfigures_v6.html)
says that image rows are listed as CC BY 2.0, but places responsibility for
per-image license verification on users.
The [CC BY 2.0 deed](https://creativecommons.org/licenses/by/2.0/) permits
commercial sharing and adaptation subject to attribution, while warning that
other rights may still apply. Accordingly, these 347 records have a matching
public page declaration, but contributor ownership, privacy/publicity rights,
and independent legal status are not authenticated. This is not a blanket
commercial/product clearance.

## Reproducibility and limits

Checker implementation and offline parser tests are in
[`audit_openimages_flickr_rights.py`](../../scripts/audit_openimages_flickr_rights.py)
and [`test_openimages_flickr_rights.py`](../../tests/test_openimages_flickr_rights.py).
The private final train-receipt JSONL SHA-256 is
`45dcaeb485cb3dc5e06c96408c2c95aa01f747117baaaae5b26ebfd42452d431`; validation
is `9afdbe3317f0c79ce052251cdbbafe76b4ffa50b191206fb6709c25a911adf62`. The
recheck was applied to saved page observations; it did not repeat page
downloads. No images were downloaded, no perceptual-hash/content-overlap test
was possible, and no model was loaded, fitted, or evaluated. The original-file
MD5 overlap remains zero for the selected 220/220 task manifests, but that is
not a substitute for perceptual deduplication.

The fixed task lists were separately filtered using only the page-declaration
gate; no performance results were consulted. The resulting private manifests
contain 176 train and 171 validation examples, with 12–18 examples per frozen
target. Their SHA-256 hashes are `71e3c95f26fdce7a5c6128e0dcbe81b47ab0f4139bf0dcc05baf341a9f374515`
and `6fbd56f0caebc77b1067dfffaadb444aaf51b4b01eabff3855c11a245f86f964`.
The page-matched pair summary hash is
`2c7c7cf3063d3fba46ec3e6d99ab1ad497f8eb848163abd7dbf9dc3d82a303d0`. The
filtering code is
[`freeze_openimages_page_matched_sample.py`](../../scripts/freeze_openimages_page_matched_sample.py);
the resulting manifests remain outside Git because they contain image IDs and
task content. Both ImageID and source-MD5 split overlaps remain zero.

Before using these records, keep the nonmatching/inaccessible entries out,
complete the project's use-specific review of the 347 page declarations, and
then acquire only a bounded approved sample for perceptual-hash and
normalized-content overlap checks. The validation pilot has 171 page-declaration
matches, so it cannot support a broad quality claim and is below the provisional
200-example floor for temporal scene/clip studies.
