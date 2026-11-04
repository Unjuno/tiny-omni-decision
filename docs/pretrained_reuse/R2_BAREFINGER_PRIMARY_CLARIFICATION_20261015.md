# Bare-Finger primary-source clarification: Audio remains unverified

After the preserved500-trial source audit and one-trial native numeric replay,
inspect the immutable Figshare26965267 version1 release notebook and complete
`figs.zip` directory. No upstream code execution, model, training or quality score.
The same bounded ZIP range reader checks member CRC and local SHA; whole ZIP
payloads/MD5 are not locally verified. Operational fetch was an inline read-only
procedure, not a new precommitted standalone producer.

Notebook `script/notebook.ipynb`:1,720,824 bytes, SHA
`1c7fa87e9678d457c636276c9168e65afdc6dcc0a3e9d31ceff66ec060e23535`.
Eight code cells again assign Kistler columns0/1 to audio and2/3 to acceleration
and call the released reader using that convention. This repeats the release's
interpretation; it supplies no acquisition wiring map or erratum resolving the
opposite assignment in the paper. Plot labels also mix kPa and V. The notebook's
epoch-like negative time offset is not applied to the pilot's918-second clocks.

The notebook JSON includes12embedded PNG outputs and14text/plain outputs.
Those encoded images were transferred with the notebook but were not decoded or
displayed. Figure-directory inspection transfers191,284 bytes and finds1505
entries/1501PNG files; no figure payloads were fetched. These are plot assets,
not native video evidence. Notebook ZIP-member transfer1,277,029 bytes; combined
bounded fetch27.468seconds. Local receipt status describes inspection only.

The author's public project page resolves HTTP200 (44,052 bytes, SHA
`e6f5c320725bd3de3e522268eab712c15caa1cd7e9cd9aae5b3a1d8de9b986f2`).
Its linked `Aightech/fingerhav` repository returns anonymous API HTTP404
(118 bytes, SHA`4f50e254a719a6b1b06b529bdaa08980e4acf0878aad0090cba7771f6ebdf0b9`).
This does not prove deletion or private status. No credentials or messages.
[Author project](https://www.aigh.tech/projects/029-fingerhav),
[fixed public release](https://doi.org/10.6084/m9.figshare.26965267.v1).

**Decision:** physical audio-channel assignment remains `UNVERIFIED_CONFLICT`.
Hold native Audio/training admission; do not select a convention using accuracy.
No claim of native video, independent synchronized audio or joint-quality gains.
Keep prior numeric/source records intact. Reopen only upon traceable primary
channel-mapping evidence, not another run on ambiguous channels.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/barefinger-primary-clarification-01/`.
`result.json` SHA`e9f0c03426acae2f5d3855bc90022873e17888de552ed20ce833ab7bd6f81d9c`;
`primary-links-receipt.json` SHA
`7eb87626f0392f65408f6240262fa0077d4206a9392199610111960339673b71`;
`figs-directory.json` SHA
`4f24ff662acade0d95aa2320d497fc9e42a35994c6e5b284fdeeb672f9e8cb97`.
This closes the current clarification attempt, not the overall Omni goal.
