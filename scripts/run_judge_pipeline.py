#!/usr/bin/env python3
"""Run Phase 2 judge scoring pipeline."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.judge import JudgePipeline
from src.logging_config import logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 judge scoring pipeline")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input inference result file path (JSONL/CSV).",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "jsonl", "csv"],
        help="Input file format, default auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory, default outputs/experiments/judge/<timestamp files>.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="all",
        choices=["jsonl", "csv", "json", "all"],
        help="Output format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of records to judge.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrent judge requests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    setup_logger(name="judge_pipeline", level=args.log_level)

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else None

    pipeline = JudgePipeline()

    try:
        results = await pipeline.run(
            input_file=input_file,
            input_format=args.input_format,
            output_dir=output_dir,
            output_format=args.output_format,
            limit=args.limit,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )

        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        print("\n" + "=" * 60)
        print("JUDGE PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total records: {total}")
        print(f"Successful: {success}")
        print(f"Failed: {failed}")

        valid_scores = [r.score for r in results if r.success and r.score is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"Average score: {avg_score:.3f}")
        print("=" * 60)
        return 0
    except Exception as exc:
        logger.error("Judge pipeline failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
