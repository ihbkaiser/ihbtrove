# IHB-Trove

Deterministic structural repair **before** conventional DataTrove filtering and deduplication for ebook, OCR, and HTML-to-text corpora. The extension uses DataTrove `PipelineStep`, `BaseFilter`, `JsonlReader`/`JsonlWriter`, `LocalPipelineExecutor`, ExactDedup, SentenceDedup, and MinHash. It runs on CPU without LLMs or embeddings.

![Comparison of DataTrove and IHB-Trove pipelines](docs/datatrove-vs-ihb-trove.png)

## Before and after on supplied books

![IHB-Trove repairs a page break and image placeholder](docs/ihb-trove-before-after-demo.png)

The left panels illustrate a **DataTrove reader/writer pass-through** without a repair step; DataTrove has no default filtering pipeline that guarantees a particular output. The right panels show structural changes verified on the supplied ebook JSONL files. The image shortens surrounding text for legibility. It does not claim to verify the factual content or recover missing OCR text. A severely corrupted third book remains excluded at the default 30% repair limit.

## Portable bundled DataTrove

This repository includes the `datatrove/` Python package and assets from the user's original `datatrove.zip` (SHA-256 `e16e9fcd796d56a8edcfebe2333073a0fae16039a03bdc7a7fce224055e63fac`). The archive is a modified DataTrove snapshot, **not** a stock PyPI wheel; its bytecode and notebook checkpoints are excluded. A single `pip install -e .` installs both `ihb_trove` and `datatrove` imports, with no separate `pip install datatrove` required. Keep this environment separate from an existing DataTrove installation because the two distributions expose the same import name.

The supplied snapshot pointed FT176 to `/workspace/storage-shared/nlp/maitn4/code/data-processing/utils/lid.176.bin` and the public suffix list to `/workspace/storage-shared/nlp/maitn4/code/data-processing/utils/public_suffix_list.dat`. These locations remain preferred if the files exist. Override them with `IHB_TROVE_FT176_MODEL` and `IHB_TROVE_PUBLIC_SUFFIX_LIST`. Otherwise FT176 falls back to the official fastText download URL/cache, and URLFilter uses tldextract's bundled suffix snapshot without a network fetch. **Model weights are not included.** The default `langdetect` gate runs offline and does not require FT176. For FT176 install `pip install -e '.[ft176]'`; for URLFilter install `pip install -e '.[url-filter]'`.

The DataTrove sources are redistributed under [Apache License 2.0](DATATROVE_LICENSE). IHB-specific changes to the bundled sources are marked in comments in `datatrove/utils/lid.py`, `datatrove/pipeline/filters/url_filter.py`, and `datatrove/pipeline/filters/language_filter.py`. The original DataTrove project is [huggingface/datatrove](https://github.com/huggingface/datatrove).

## Folder in → `survive/` and `eliminated/` out

Requires Python 3.11+. The package declares the Vietnamese tokenizer and JSONL dependencies directly; its default `langdetect` language gate works offline without a model download.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/ihb-trove /path/to/input_jsonl_folder /path/to/new_output_folder
```

If a run is interrupted, resume it with the same input, output, and pipeline options:

```bash
.venv/bin/ihb-trove /path/to/input_jsonl_folder /path/to/new_output_folder --resume
```

The runner checkpoints completed stages under `OUTPUT/.ihb_trove_cache/`. The cache is retained after interruption and removed after a successful run. A resumed run reuses completed stages and reruns only the unfinished stage; if interrupted during final routing, it rebuilds `survive/` and `eliminated/` from cached stage outputs. Input content and pipeline options must match the checkpoint. Resume re-reads the input to verify its content fingerprint. Legacy interrupted runs that only have an `ihb-trove-*` temporary directory cannot be resumed by this feature.

By default, the CLI detects the available CPU quota, uses up to 32 workers and creates up to four DataTrove shards per worker, capped by the number of JSONL files. Task count is always capped by the file count because this runner shards its reader by file. Input validation/staging reads up to 16 files concurrently. Pass `--tasks` and `--workers` to override these defaults. Set `--workers 1` for a serial baseline. Progress bars report staged files, DataTrove reader file shards, and records being routed; bars are disabled automatically when output is not an interactive terminal.

The run writes a live orchestration log to `OUTPUT/logs/run.log`; DataTrove executor/task logs remain under `OUTPUT/logs/datatrove/` after completion. Follow the live log in another terminal with:

```bash
tail -f /path/to/new_output_folder/logs/run.log
```

To confirm both import names resolve to this checkout:

```bash
.venv/bin/python -c "import datatrove, ihb_trove; print(datatrove.__file__, ihb_trove.__file__)"
```

The script scans `*.jsonl` **recursively**, processes all documents together for global deduplication, and mirrors each input file's relative path under both outputs. Each source has a corresponding JSONL file in each output, possibly empty. Every input record goes to exactly one side. Invalid JSON and records without text go to `eliminated/` with a reason; filtering and whole-document dedup removals also go there. SentenceDedup edits a surviving document's text without routing its original copy to `eliminated/`.

```text
input_jsonl_folder/
  book_a.jsonl
  collection/book_b.jsonl

new_output_folder/
  survive/book_a.jsonl
  survive/collection/book_b.jsonl
  eliminated/book_a.jsonl
  eliminated/collection/book_b.jsonl
  .ihb_trove_cache/              # retained only while a run is incomplete
  summary.json
```

Every eliminated record carries `metadata.filter_reason` and `metadata.ihb_exclusion_stage`; records retain their source file and one-based source line in metadata. The CLI refuses a nonempty output folder and rejects overlapping input/output directories. Intermediate DataTrove signatures live in a temporary work directory and are removed after a successful run; persistent run and executor logs stay under `output/logs/`.

```bash
# If the corpus is mixed media and real image references should be retained:
.venv/bin/ihb-trove INPUT OUTPUT --keep-image-placeholders

# Exploratory rescue of a heavily corrupted OCR book; review before training:
.venv/bin/ihb-trove INPUT OUTPUT --max-repair-fraction 0.40

# Custom page marker in addition to form feed (repeat the option for multiple markers):
.venv/bin/ihb-trove INPUT OUTPUT --page-marker='[PAGE]'
```

The default repair removal limit is **30%**. For this text-only corpus, Markdown/HTML image references and generated unavailable-image notes are removed by default. `--language` defaults to `vi`; `--language-backend langdetect` is deterministic and offline. `ft176` and `glotlid` use DataTrove's native `LanguageFilter` and require their models to be available. `--skip-long-line-loops` disables the conservative exact OCR word-loop repair.

## Processing order

1. `UnicodeNormalizer` → `ImagePlaceholderRepair` → `PageStructureRepair` → `LocalRepeatedSpanRepair` → `LongLineLoopRepair` → `RepairMetrics`.
2. `RepairQualityFilter` → offline language filter (or DataTrove `LanguageFilter`) → `GopherRepetitionFilter` → `BookQualityFilter`.
3. Separate DataTrove jobs: **ExactDedup → SentenceDedup → MinHash**. A disk-backed routing pass compares stage outputs by a stable source-record ID so every dropped document receives a reason, including whole-document dedup drops.

`PageStructureRepair` recognizes `---`, form feed, and configured markers. It learns recurring edge lines with normalized signatures and RapidFuzz OCR clustering; stable numeric titles can be headers, while changing recipe numbers are preserved. Page numbers require a consistent offset, including contiguous offset changes. `LocalRepeatedSpanRepair` verifies rolling-hash hits on nearby long multiline spans. `LongLineLoopRepair` keeps one copy of a substantial word span repeated at least four times within a long line. Short repeated headings such as `Nguyên liệu` remain content. All repair counters, before/after lengths, removed fraction, confidence, and version are written to `document.metadata["repair"]`.

## Validation

```bash
.venv/bin/python -m pytest -q
```

On the three supplied JSONL books, the default gate produced **2 surviving and 1 eliminated** document. The eliminated power book had 32.7% of its characters removed during exact OCR repair and was marked `excessive_repair`. With an explicit 40% exploratory limit, all three documents survived through ExactDedup, SentenceDedup, and MinHash. The six Markdown image references and one generated image note in this pool were removed; none remains in the final text. These are corpus checks, not evidence that the remaining OCR prose or medical claims are correct.

The per-document repair steps hold a document in memory. For very large books, split the input into chapters or volumes with intact page markers. Global dedup quality cannot be estimated from only three different books. Existing table-of-contents and publisher back matter are intentionally left for a separate corpus policy.
