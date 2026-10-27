# Existing Open Images train label inventory

Before reading labels, freeze the existing 176 page-declaration-matched train
tasks, SHA-256 `71e3c95f26fdce7a5c6128e0dcbe81b47ab0f4139bf0dcc05baf341a9f374515`,
and the previously frozen 11-class scope. Stream the pinned complete human
train label source and verify its existing manifest SHA-256. For these fixed
image IDs only, record all in-scope verified positives and negatives; require
the original target to remain a verified positive. Count existing image IDs
with at least three in-scope negatives, by target and by whether a frozen
VideoPrism train feature already exists. Do not inspect validation, select a
new subset for evaluation, or fit a model. No image is newly downloaded.

The page-declaration checks do not establish copyright ownership or current
media availability, nor does a count prove safe dataset splits or a balanced
choice distribution. Inventory records are private local artifacts. A future
training set requires a separately frozen sampling rule and source integrity
checks; preserve the original train and validation artifacts unchanged.
