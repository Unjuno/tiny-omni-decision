# ADL 25-activity partial inventory: completed source screen

The fixed [inventory protocol](R2_ADL_PARTIAL_INVENTORY_PROTOCOL_20261027.md)
finished in 1,068.78 seconds. All 25 archive names were accounted for: two
previously fully MD5-verified activities, 18 first-sensor members checked by
ZIP CRC and local SHA-256, and five `UNSUPPORTED_SIZE` activities whose
selected members exceeded the preregistered bounds. The scan did not establish
whether those five activities have split coverage. Their names are
`00_no-activity`, `01_walk`, `02_walk-to-room`, `15_vacuum-cleaning` and
`24_wipe-stove`. Do not interpret `UNSUPPORTED_SIZE` as a source-data error.

For the 20 inspected activities, every first sensor had at least one event in
each fixed split and none had an unassigned row. They contain 5,378 train and
1,707 validation activity rows in total; those are repeated events from a small
number of people in one environment, not independent people or sessions. Every
inspected activity has only **two validation participants**. Two classes are
especially sparse: sweep has 6 train/3 validation events and typing has 10/4.
Having nonempty splits is not a statistical adequacy test. Do not silently
remove those classes or duplicate them to improve a score.

For the 18 partial activities the complete publisher archive MD5 was not
verified; ZIP CRC and locally computed member SHA are weaker evidence, and
the other four sensors were not checked. Only two full archives have all five
sensor identities checked. Neither this scan nor the separate two-activity
exact waveform check proves session separation, temporal alignment, partial
media nonoverlap, label quality, or a task requiring multiple inputs. The
special no-participant activity is one of the five uninspected activities.
The source has no released video, and its 14 participants in one environment
cannot by themselves meet the provisional 200-independent-group screen or the
full multimodal objective. **The source remains unadmitted for training and
quality evaluation.**

The terminal receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/adl-partial-inventory-20261027-attempt-01/result.json`,
SHA-256 `39c8e321e0785990a5f557672668ac36f7bf73ae3cf8abe8bb73ee2efa14edf6`.
It retains every member digest, each count, unsupported sizes, fixed protocol
and source hashes. No raw media, model score or sealed audit was produced. The
live process finished normally; `operation.json` was a startup record and its
old `running` field is not its terminal status. Further work needs an adequately
independent, rights-cleared both-input task and complete source-integrity checks;
this inventory alone does not justify a long fit or product promotion.
