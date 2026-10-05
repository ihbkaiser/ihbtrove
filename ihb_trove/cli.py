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
    parser.add_argument("--language", default="vi")
    parser.add_argument("--language-backend", choices=("langdetect", "ft176", "glotlid"), default="langdetect")
    parser.add_argument("--page-marker", action="append", dest="page_markers")
    parser.add_argument("--max-repair-fraction", type=float, default=0.30)
    parser.add_argument("--keep-image-placeholders", action="store_true")
    parser.add_argument("--skip-long-line-loops", action="store_true")
    args = parser.parse_args()
    summary = run_folder_pipeline(
        args.input_folder,
        args.output_folder,
        tasks=args.tasks,
        workers=args.workers,
        language=args.language,
        language_backend=args.language_backend,
        page_markers=tuple(args.page_markers or ("---",)),
        max_repair_fraction=args.max_repair_fraction,
        remove_image_placeholders=not args.keep_image_placeholders,
        repair_long_line_loops=not args.skip_long_line_loops,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
