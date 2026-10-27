# Train quartet source candidates for rights screening

Freeze this bounded exploratory selection before reading source records. Stream
and hash the pinned full train-label and train-metadata objects, using only the
11-class scope, explicit verified positives and three distinct verified negatives.
Exclude IDs, original MD5s and normalized author identities from the pinned
official validation metadata and from both earlier train candidate receipt sets.
Exclude the IDs of the previously fixed 176 train tasks as well. Source metadata
must declare train and CC BY 2.0 and have complete attribution fields.

Sort remaining images by SHA-256 `quartet-unique-v1|17|<image ID>` and retain
only the first image per normalized Flickr author and original MD5. Apply the
existing fixed disjoint same-option quartet construction. For all generated
quartets, count target availability by class, sort quartets by the minimum of
their four class counts (ascending), breaking ties by SHA-256 of
`quartet-screen-v1|17|<sorted four image IDs>`. Select at most 64 quartets for
page screening; report all counts and absent classes rather than refill or alter
selection after observing page results. This deliberately favors scarce targets
based only on train labels and is not a naturally distributed benchmark.

Save only selected candidate labels, group definitions and private attribution
receipts outside the repository after both source hashes pass. Check the Flickr
page declarations for those candidates; do not download media or fit a model in
this procedure. Report how many complete quartets survive page checks without
replacing failed images. A page match is not ownership proof; pretraining
overlap, image-content duplication, and independent evaluation remain open.
Existing evaluation data, tasks, and artifacts remain unchanged.
