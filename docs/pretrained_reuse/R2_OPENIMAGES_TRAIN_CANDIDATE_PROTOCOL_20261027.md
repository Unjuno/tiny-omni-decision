# New in-scope train image candidates

Freeze this candidate selection before scanning records. Use the previously
verified train annotation object, train image metadata, 11-class scope, and
existing 176-image train manifest. Exclude every existing train image ID and
every ID, source MD5 and normalized Flickr author identity in the pinned
official validation image metadata (metadata only, no validation annotations).
Require train metadata, CC BY 2.0 declaration, attribution fields, a verified
in-scope positive and at least three distinct in-scope verified negatives.

Generate all eligible `(image ID, positive target)` pairs. Sort by the SHA-256
of `openimages-in-scope-candidate-v1|17|<image ID>|<target label>` and greedily
accept at most 30 unique images per target, never reusing an image ID, source
MD5 or normalized author across the selected candidates. This is an inventory
for rights checks, not a training sample or a role-balanced task. Report all
per-target shortfalls rather than changing this rule after seeing the output.

Require source response generation, ETag, length and SHA-256; save private
candidate labels and attribution receipts only after both streams pass.
Then check landing-page declarations with existing code; no image media,
validation labels, test labels or models are loaded. A matching declaration is
not proof of ownership. No candidate is admitted for training before rights,
asset identity, independent split, option-balance and task design gates pass.
