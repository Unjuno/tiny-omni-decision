# BeDAViN source screen: raw audio is not a paired AV dataset

## Decision

**REVIEW for a possible future audio-only research lead; not admitted as a
paired audio/video source or product-quality benchmark.** The official Zenodo
record describes 135 manually recorded sound-event files, over two hours,
with 24 categories and 96 kHz/24-bit audio. Its description states CC BY 4.0
for that released dataset. The record lists split archives totaling 6.8 GB,
which cannot fit the current approximately 1.2 GiB free C: capacity. This is a
source description, not validation of the files or class labels.

The official research paper describes a larger dataset of 2,258 audio samples
and 10.8 hours. The pinned project README explains that the complete framework
combines manually recorded sounds with AudioSet and Freesound sources, while
its audiovisual navigation uses separately acquired Matterport3D scenes,
SoundSpaces impulse responses and simulation. The paper's 2,258 samples must
not be conflated with the 135 raw recordings on Zenodo. The README explicitly
places Matterport3D-derived task data and models under separate terms and
CC BY-NC-SA 3.0 US; the Matterport project requires a signed access agreement.
The raw audio release does not itself provide synchronized real video for the
required joint-input test. No claim is made that the other audio components
inherit the raw-recording license.

A direct metadata.csv fetch returned HTTP 429 during this screen. It was not
retried through another route; file contents, source identities, train/validation
splits, independence, event semantics, class balance and exact licensing of each
component remain unverified. No files, media, model weights, benchmark test data,
access agreement or paid service were used. Reconsider audio-only use only after
metadata and component-level provenance are verified, a task and group split
are frozen, and storage permits; this cannot by itself satisfy the user's
simultaneous audiovisual or 95%+ general Decision goal. Do not use BeDAViN as a
reason to drop the existing source semantics gate.

## Primary evidence

- [Official Zenodo v1 record, 135 manual recordings and raw archive](https://zenodo.org/records/14486235)
- [Pinned ENMuS README, scene and component rights](https://github.com/ZhanboShiAI/ENMuS/blob/2cf856cd4c8c604edee232c1180e299ca96a112c/README.md)
- [Matterport3D official access page](https://niessner.github.io/Matterport/)
- [AAAI paper abstract, 2,258 audio samples](https://ojs.aaai.org/index.php/AAAI/article/view/33608)
