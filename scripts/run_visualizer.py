#!/usr/bin/env python3
"""Run Module B: visualization for Phase 4 analysis CSV outputs."""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logging_config import logger, setup_logger
from src.stats import StatsVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 visualization generator")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Primary Phase 4 CSV file (valid_scored_records/group_metrics/final_report).",
    )
    parser.add_argument(
        "--valid-records-file",
        type=str,
        default="",
        help="Optional valid_scored_records CSV when --input-file has no raw score column.",
    )
    parser.add_argument(
        "--group-metrics-file",
        type=str,
        default="",
        help="Optional group_metrics CSV for delta bar chart mean values.",
    )
    parser.add_argument(
        "--arm-metrics-file",
        type=str,
        default="",
        help="Optional objective arm_metrics CSV for dose-response trend chart.",
    )
    parser.add_argument(
        "--category-arm-metrics-file",
        type=str,
        default="",
        help="Optional category_arm_metrics CSV for heterogeneity grouped bar chart.",
    )
    parser.add_argument(
        "--cross-model-summary-file",
        type=str,
        default="",
        help="Optional cross_model_summary CSV for cross-model overview charts.",
    )
    parser.add_argument(
        "--cross-model-factor-metrics-file",
        type=str,
        default="",
        help="Optional cross_model_factor_metrics CSV for cross-model heatmap charts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments/figures",
        help="Directory to save figures and summary JSON.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated output formats. Supported: png,pdf.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution in DPI.")
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
    setup_logger(name="visualizer", level=args.log_level)

    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    visualizer = StatsVisualizer(dpi=args.dpi)

    try:
        saved = visualizer.run(
            input_file=Path(args.input_file),
            output_dir=Path(args.output_dir),
            formats=formats,
            valid_records_file=Path(args.valid_records_file) if args.valid_records_file else None,
            group_metrics_file=Path(args.group_metrics_file) if args.group_metrics_file else None,
            arm_metrics_file=Path(args.arm_metrics_file) if args.arm_metrics_file else None,
            category_arm_metrics_file=(
                Path(args.category_arm_metrics_file) if args.category_arm_metrics_file else None
            ),
            cross_model_summary_file=(
                Path(args.cross_model_summary_file) if args.cross_model_summary_file else None
            ),
            cross_model_factor_metrics_file=(
                Path(args.cross_model_factor_metrics_file) if args.cross_model_factor_metrics_file else None
            ),
        )
        print("\n" + "=" * 64)
        print("VISUALIZATION COMPLETED")
        print("=" * 64)
        for k, v in saved.items():
            print(f"{k}: {v}")
        print("=" * 64)
        return 0
    except Exception as exc:
        logger.error("Visualization failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
