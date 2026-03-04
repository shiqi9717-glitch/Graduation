#!/usr/bin/env python3
"""Run Phase 4 analysis on judge results."""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import logger, setup_logger
from src.stats import StatsAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 statistics analyzer")
    parser.add_argument("--input-file", type=str, required=True, help="Judge output file (JSONL/CSV).")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "jsonl", "csv"],
        help="Input format (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: data/results/analysis).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logger(name="analysis", level=args.log_level)

    analyzer = StatsAnalyzer()
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        saved = analyzer.run(
            input_file=Path(args.input_file),
            input_format=args.input_format,
            output_dir=output_dir,
        )
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        for k, v in saved.items():
            print(f"{k}: {v}")
        print("=" * 60)
        return 0
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

