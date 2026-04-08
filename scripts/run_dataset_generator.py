#!/usr/bin/env python3
"""Run Module A: automatic dataset generator."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DatasetGenerator
from src.logging_config import logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sycophancy dataset via LLM")
    parser.add_argument("--num-questions", type=int, default=300, help="Total number of questions.")
    parser.add_argument(
        "--questions-per-request",
        type=int,
        default=10,
        help="How many items to request per API call.",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent API calls per round.")
    parser.add_argument("--max-rounds", type=int, default=40, help="Max generation rounds.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument(
        "--seed-topic",
        type=str,
        default="moral dilemmas and business decisions",
        help="Primary generation topic.",
    )
    parser.add_argument(
        "--domain-hint",
        type=str,
        default="business, ethics, compliance, workplace politics",
        help="Domain coverage hint.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/raw/expanded_sycophancy_database.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="",
        help="Optional summary JSON path.",
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
    setup_logger(name="dataset_generator", level=args.log_level)

    generator = DatasetGenerator()

    try:
        summary = await generator.generate(
            num_questions=args.num_questions,
            output_file=Path(args.output_file),
            questions_per_request=args.questions_per_request,
            concurrency=args.concurrency,
            max_rounds=args.max_rounds,
            temperature=args.temperature,
            seed_topic=args.seed_topic,
            domain_hint=args.domain_hint,
            summary_file=Path(args.summary_file) if args.summary_file else None,
        )
        print("\n" + "=" * 64)
        print("DATASET GENERATION COMPLETED")
        print("=" * 64)
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=" * 64)
        return 0
    except Exception as exc:
        logger.error("Dataset generation failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

