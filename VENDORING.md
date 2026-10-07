# Bundled DataTrove provenance

`datatrove/` is copied from the user-provided `datatrove.zip` rather than from a
stock upstream release. SHA-256 of the input archive:

`e16e9fcd796d56a8edcfebe2333073a0fae16039a03bdc7a7fce224055e63fac`

The archive contains local modifications. The original snapshot included
absolute paths for `lid.176.bin` and `public_suffix_list.dat`, and a
`LangDetectLID` backend. Those paths are retained as preferred sources when
present. IHB-Trove added explicit environment overrides and portable fallback
behavior in `datatrove/utils/lid.py` and
`datatrove/pipeline/filters/url_filter.py`. It also fixed a `glotlid` backend
assignment in `datatrove/pipeline/filters/language_filter.py`. The Gopher
repetition filter also has an optional no-tokenizer path for callers that use
only language-neutral character checks. These files contain inline change
notices or comments identifying the IHB-specific changes.

Python bytecode and notebook checkpoints were omitted from the archive import.
The DataTrove blacklist and tokenizer assets are included. The FastText model
and the local public suffix list are not included. This repository has no
claim of being identical to any official DataTrove release.

Upstream: https://github.com/huggingface/datatrove

License for bundled DataTrove: [Apache License 2.0](DATATROVE_LICENSE).
