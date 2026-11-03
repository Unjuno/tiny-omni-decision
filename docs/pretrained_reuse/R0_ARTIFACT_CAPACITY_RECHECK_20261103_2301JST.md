# R0 post-build capacity sample — 2026-11-03 23:01 JST

The user-owned Quartus build (PID 28280) was no longer present in the process
table after the 22:59 JST check. I did not stop or modify it. Three C: free
space samples were taken 22 seconds apart:

| Local time | Free bytes |
|---|---:|
| 23:01:09 | 1,759,301,632 |
| 23:01:31 | 1,767,297,024 |
| 23:01:53 | 1,766,825,984 |
| Mean | 1,764,474,880 |

The mean is **365.18 MiB below** the feature extractor's immutable 2 GiB
reserve. The 44-second three-sample spread is 7.63 MiB, so capacity is not
treated as restored. No Python, FFmpeg, or download process was active. No
attempt was made to remove the two exact-prefix partials after automatic action
review rejected that operation; both remain and zero bytes were freed.

The fixed RoboMNIST cache remains at **1,778 / 1,780** group directories,
`INCOMPLETE`, with the last run metadata update at 22:26:52 JST. Its two missing
groups are frozen validation groups. No readout was fitted, no validation rows
were omitted, and no split/config/hash was changed. Resume only when stable
capacity is restored above the existing 2 GiB guard. All source media, feature
receipts, model/checkpoint artifacts, and user-owned work remain untouched.
