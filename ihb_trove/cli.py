"""Command line interface for recursive IHB-Trove JSONL filtering."""

import argparse
import json
from pathlib import Path

from .folder_pipeline import run_folder_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="One or more input folders followed by the output folder",
    )
    parser.add_argument(
        "--input-folders-file",
        type=Path,
        help="Text file with one input folder per line; relative paths use the list file's directory",
    )
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
        "--dedup-finder-workers",
        type=int,
        default=None,
        help="Hash-range workers for Exact/SentenceDedup finder stages (default: up to 16)",
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
    output_folder = args.paths[-1]
    input_folders = list(args.paths[:-1])
    if args.input_folders_file is not None:
        list_path = args.input_folders_file.expanduser().resolve()
        try:
            lines = list_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            parser.error(f"cannot read input folders file {list_path}: {exc}")
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            folder = Path(value).expanduser()
            if not folder.is_absolute():
                folder = list_path.parent / folder
            input_folders.append(folder)
    if not input_folders:
        parser.error("provide at least one input folder, either before the output folder or in --input-folders-file")
    summary = run_folder_pipeline(
        input_folders[0] if len(input_folders) == 1 else input_folders,
        output_folder,
        tasks=args.tasks,
        workers=args.workers,
        dedup_finder_workers=args.dedup_finder_workers,
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
