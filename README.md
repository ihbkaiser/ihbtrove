# IHB-Trove

Deterministic structural repair **before** conventional DataTrove filtering and deduplication for ebook, OCR, and HTML-to-text corpora. The extension uses DataTrove `PipelineStep`, `BaseFilter`, `JsonlReader`/`JsonlWriter`, `LocalPipelineExecutor`, ExactDedup, SentenceDedup, and MinHash. It does not modify DataTrove core and runs on CPU without LLMs or embeddings.

![Comparison of DataTrove and IHB-Trove pipelines](docs/datatrove-vs-ihb-trove.png)

## Folder in → `survive/` and `eliminated/` out

Requires Python 3.11+ and DataTrove 0.10.1. The package declares the Vietnamese tokenizer and JSONL dependencies directly; its default `langdetect` language gate works offline without a model download.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/ihb-trove /path/to/input_jsonl_folder /path/to/new_output_folder --tasks 4 --workers 4
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
  summary.json
```

Every eliminated record carries `metadata.filter_reason` and `metadata.ihb_exclusion_stage`; records retain their source file and one-based source line in metadata. The CLI refuses a nonempty output folder and rejects overlapping input/output directories. Intermediate DataTrove signatures and logs live in a temporary work directory and are removed after a successful run.

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
