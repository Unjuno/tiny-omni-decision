# Train-only Open Images symmetric-choice bounded prefix screen

Before reading labels, freeze this feasibility screen. Use the exact official
Open Images V6 train human-label object generation `1580742923196487` and ETag
`a0c8491396bfa736bec85cb172180bd3`, and only the first 67,108,864 bytes via
HTTP range. Require HTTP 206, exact Content-Range and matching ETag; fail closed
on a missing or changed header. Keep a SHA-256 of received bytes and discard the
incomplete final CSV row and final image group. Do not load validation, test,
media, predictions or model weights. This lexicographic-prefix sample is not a
random or representative corpus estimate, and does not verify the SHA-256 of
the full 2.47 GB source; report its integrity limitation explicitly.

Using the 11 classes already frozen from train support, count for each class
unique image groups with a `verification` positive and at least three distinct
`verification` negatives within the other 10 classes. Require explicit
negative labels; never infer absence. Also count available negative classes per
positive and contradiction failures. Do not drop classes, select examples,
create a train/eval split, download images, fit a model, or tune to observed
validation. Compare this bounded feasibility with the prior hierarchy screen
without treating missing data as zero support in the full corpus. Historical
CC-BY metadata and individual Flickr rights checks are not independently
revalidated in this prefix screen. A later frozen, full-source audit and role-
symmetric task design with provenance and group integrity are required.
