"""Run IHB-Trove and optional DataTrove dedup jobs on local JSONL files."""

import argparse
from glob import escape
from pathlib import Path

from ihb_trove.pipelines import (
    build_book_pipeline,
    build_exact_dedup_pipeline,
    build_minhash_pipeline,
    build_sentence_dedup_pipeline,
)


def has_records(folder: Path) -> bool:
    return any(path.stat().st_size > 0 for path in folder.glob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A JSONL file or directory of JSONL files")
    parser.add_argument("output", type=Path, help="Fresh output directory for the run")
    parser.add_argument("--stage", choices=("repair", "exact", "sentence", "minhash", "all"), default="repair")
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--dedup-tokenizer-language",
        default=None,
        help="Optional DataTrove tokenizer override; defaults to language-neutral Unicode tokenization",
    )
    parser.add_argument("--keep-image-placeholders", action="store_true")
    parser.add_argument("--skip-long-line-loops", action="store_true")
    args = parser.parse_args()
    if args.tasks < 1 or args.workers < 1:
        parser.error("tasks and workers must be positive")
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")

    source = args.input.parent if args.input.is_file() else args.input
    glob_pattern = escape(args.input.name) if args.input.is_file() else "*.jsonl"
    root = args.output.resolve()
    book = build_book_pipeline(
        source,
        root,
        glob_pattern=glob_pattern,
        tasks=args.tasks,
        workers=args.workers,
        remove_image_placeholders=not args.keep_image_placeholders,
        repair_long_line_loops=not args.skip_long_line_loops,
    )
    book.run()
    if args.stage == "repair" or not has_records(root / "filtered"):
        print(f"Filtered documents: {root / 'filtered'}")
        return

    exact = build_exact_dedup_pipeline(root / "filtered", root, tasks=args.tasks, workers=args.workers)
    exact.run()
    if args.stage == "exact":
        return
    if not has_records(root / "exact"):
        print("No documents survived exact deduplication.")
        return
    sentence = build_sentence_dedup_pipeline(
        root / "exact",
        root,
        tasks=args.tasks,
        workers=args.workers,
        tokenizer_language=args.dedup_tokenizer_language,
    )
    sentence.run()
    if args.stage == "sentence":
        return
    if not has_records(root / "sentence"):
        print("No documents survived sentence deduplication.")
        return
    minhash = build_minhash_pipeline(
        root / "sentence",
        root,
        tasks=args.tasks,
        workers=args.workers,
        tokenizer_language=args.dedup_tokenizer_language,
    )
    minhash.run()
    print(f"Final documents: {root / 'minhash'}")


if __name__ == "__main__":
    main()
