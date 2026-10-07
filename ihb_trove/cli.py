"""Command line interface for recursive IHB-Trove JSONL filtering."""

import argparse
import json
from pathlib import Path

from .folder_pipeline import run_folder_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="DataTrove shards (default: up to 4 per worker; always capped by JSONL file count)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent DataTrove workers (default: CPU quota and file count, capped at 32)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed stage checkpoints in the existing output folder",
    )
    parser.add_argument(
        "--dedup-tokenizer-language",
        default=None,
        help="Optional DataTrove tokenizer override for SentenceDedup/MinHash; default is language-neutral Unicode",
    )
    parser.add_argument("--page-marker", action="append", dest="page_markers")
    parser.add_argument("--keep-image-placeholders", action="store_true")
    parser.add_argument("--skip-long-line-loops", action="store_true")
    parser.add_argument(
        "--dehyphenate-line-breaks",
        action="store_true",
        help="Join visible hyphenated line breaks (opt-in; may alter legitimate words and names)",
    )
    parser.add_argument("--max-consecutive-blank-lines", type=int, default=2)
    args = parser.parse_args()
    summary = run_folder_pipeline(
        args.input_folder,
        args.output_folder,
        tasks=args.tasks,
        workers=args.workers,
        dedup_tokenizer_language=args.dedup_tokenizer_language,
        page_markers=tuple(args.page_markers or ("---",)),
        remove_image_placeholders=not args.keep_image_placeholders,
        repair_long_line_loops=not args.skip_long_line_loops,
        dehyphenate_line_breaks=args.dehyphenate_line_breaks,
        max_consecutive_blank_lines=args.max_consecutive_blank_lines,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
