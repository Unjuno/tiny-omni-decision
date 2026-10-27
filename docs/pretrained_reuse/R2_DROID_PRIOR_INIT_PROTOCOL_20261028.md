# DROID nested multimodal probe with prior initialization

Repeat the [nested multimodal protocol](R2_DROID_NESTED_MULTIMODAL_PROTOCOL_20261028.md)
with identical source, folds, feature transforms, optimizer and arms. Replace
the randomly initialized linear weights with zeros and initialize each bias to
the logarithm of the class frequency computed exclusively on that fit's train
rows. The initial prediction is therefore the fit-specific class prior.
Evaluate the fixed candidate updates `0, 1, 10, 50, 100, 200` on the inner
folds and refit at the selected update on outer training rows. This jointly
tests prior-based initialization and inclusion of the zero-update baseline;
it does not isolate one from the other. Each inner and outer fit computes its
own prior, without using its held labels. No validation or sealed audit is
opened or scored. Save all selected updates and OOF results.

The cohort and earlier outcomes are already observed; this is development
evidence, not a blind estimate. No source, label, class or option is removed.
