#!/usr/bin/env python3
"""Run end-to-end pipeline: perturbation -> inference -> judge -> analysis."""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.inference_settings import InferenceSettings
from src.data_pipeline import DataPerturbationPipeline
from src.inference import InferencePipeline, ModelConfig, ModelProvider, QuestionType
from src.judge import JudgePipeline
from src.logging_config import logger, setup_logger
from src.stats import StatsAnalyzer


def _detect_format(file_path: Path, mode: str = "auto") -> str:
    if mode != "auto":
        return mode
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported file format: {file_path}")


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


async def _run_inference(
    input_file: Path,
    input_format: str,
    question_column: str,
    question_type: str,
    model_name: str,
    provider: str,
    api_key: str,
    api_base: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    batch_size: int,
    concurrency: int,
    output_dir: Path,
    output_format: str,
) -> Path:
    model_default = InferenceSettings.DEFAULT_MODELS.get(model_name)
    timeout = getattr(model_default, "timeout", 30) if model_default else 30
    max_retries = getattr(model_default, "max_retries", 3) if model_default else 3

    model_config = ModelConfig(
        provider=ModelProvider(provider),
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    pipeline = InferencePipeline(model_config)
    await pipeline.initialize()
    try:
        if input_format == "jsonl":
            questions = pipeline.load_questions_from_jsonl(
                input_file, QuestionType(question_type), limit=None
            )
        elif input_format == "csv":
            questions = pipeline.load_questions_from_csv(
                input_file, question_column, QuestionType(question_type), limit=None
            )
        else:
            raise ValueError(f"Unsupported inference input format: {input_format}")

        if not questions:
            raise ValueError("No questions loaded for inference.")

        requests = pipeline.create_inference_requests(
            questions, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        )
        all_results = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            batch_result = await pipeline.run_batch(batch, concurrency_limit=concurrency)
            all_results.extend(batch_result.responses)

        saved = pipeline.save_results(output_dir=output_dir, format=output_format)
        if "jsonl" in saved:
            return saved["jsonl"]
        if "csv" in saved:
            return saved["csv"]
        if "json" in saved:
            return saved["json"]
        raise RuntimeError("Inference outputs were not saved.")
    finally:
        await pipeline.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full sycophancy evaluation pipeline")

    # Inputs
    parser.add_argument("--raw-input-file", type=str, default="", help="Raw question file for perturbation step.")
    parser.add_argument("--raw-input-format", type=str, default="auto", choices=["auto", "jsonl", "csv"])
    parser.add_argument(
        "--inference-input-file",
        type=str,
        default="",
        help="Direct inference input file (used when --skip-perturbation).",
    )
    parser.add_argument(
        "--inference-results-file",
        type=str,
        default="",
        help="Existing inference result file (used when --skip-inference).",
    )
    parser.add_argument(
        "--judge-results-file",
        type=str,
        default="",
        help="Existing judge result file (used when --skip-judge).",
    )

    # Skips
    parser.add_argument("--skip-perturbation", action="store_true", help="Skip data perturbation step.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference step.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge step.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis step.")

    # Inference options
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "deepseek", "qwen", "custom"],
    )
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--question-column", type=str, default="question")
    parser.add_argument(
        "--question-type",
        type=str,
        default="sycophancy",
        choices=["sycophancy", "control", "perturbed"],
        help="Fallback question type used by inference module.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)

    # Outputs
    parser.add_argument("--output-root", type=str, default="data/results/full_pipeline")
    parser.add_argument(
        "--output-format", type=str, default="all", choices=["jsonl", "csv", "json", "all"]
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    setup_logger(name="full_pipeline", level=args.log_level)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    perturbation_dir = output_root / "perturbation"
    inference_dir = output_root / "inference"
    judge_dir = output_root / "judge"
    analysis_dir = output_root / "analysis"

    logger.info("Full pipeline output root: %s", output_root)

    perturbation_output: Optional[Path] = None
    inference_output: Optional[Path] = None
    judge_output: Optional[Path] = None
    analysis_outputs: Dict[str, Path] = {}

    try:
        # Step 0: perturbation
        if args.skip_perturbation:
            logger.info("Skip perturbation step by flag.")
        else:
            if not args.raw_input_file:
                raise ValueError("raw-input-file is required when perturbation is not skipped.")
            raw_input = Path(args.raw_input_file)
            perturbation_dir.mkdir(parents=True, exist_ok=True)
            perturbation_output = perturbation_dir / "perturbed_questions.csv"

            logger.info("Step 0/3: running data perturbation...")
            dp = DataPerturbationPipeline()
            await dp.run_pipeline(
                input_path=raw_input,
                output_path=perturbation_output,
                sample_size=None,
            )
            logger.info("Perturbation output: %s", perturbation_output)

        # Step 1: inference
        if args.skip_inference:
            if not args.inference_results_file:
                raise ValueError("inference-results-file is required when skip-inference is set.")
            inference_output = Path(args.inference_results_file)
            logger.info("Skip inference; using existing file: %s", inference_output)
        else:
            if args.skip_perturbation:
                if not args.inference_input_file:
                    raise ValueError(
                        "inference-input-file is required when skip-perturbation is set."
                    )
                inference_input = Path(args.inference_input_file)
            else:
                inference_input = perturbation_output  # type: ignore[assignment]

            input_fmt = _detect_format(inference_input, "auto")
            inference_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Step 1/3: running model inference...")
            inference_output = await _run_inference(
                input_file=inference_input,
                input_format=input_fmt,
                question_column=args.question_column,
                question_type=args.question_type,
                model_name=args.model,
                provider=args.provider,
                api_key=args.api_key,
                api_base=args.api_base,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                system_prompt=args.system_prompt,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                output_dir=inference_dir,
                output_format=args.output_format,
            )
            logger.info("Inference output: %s", inference_output)

        # Step 2: judge
        if args.skip_judge:
            if not args.judge_results_file:
                raise ValueError("judge-results-file is required when skip-judge is set.")
            judge_output = Path(args.judge_results_file)
            logger.info("Skip judge; using existing file: %s", judge_output)
        else:
            if inference_output is None:
                raise RuntimeError("No inference output available for judge step.")
            judge_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Step 2/3: running judge scoring...")
            jp = JudgePipeline()
            await jp.run(
                input_file=inference_output,
                input_format=_detect_format(inference_output, "auto"),
                output_dir=judge_dir,
                output_format=args.output_format,
                limit=None,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
            )
            judge_output = _latest_file(judge_dir, "judge_results_*.jsonl")
            if judge_output is None:
                judge_output = _latest_file(judge_dir, "judge_results_*.csv")
            if judge_output is None:
                raise RuntimeError("Could not find judge output file.")
            logger.info("Judge output: %s", judge_output)

        # Step 3: analysis
        if args.skip_analysis:
            logger.info("Skip analysis step by flag.")
        else:
            if judge_output is None:
                raise RuntimeError("No judge output available for analysis step.")
            analysis_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Step 3/3: running statistical analysis...")
            analyzer = StatsAnalyzer()
            analysis_outputs = analyzer.run(
                input_file=judge_output,
                input_format=_detect_format(judge_output, "auto"),
                output_dir=analysis_dir,
            )
            logger.info("Analysis outputs: %s", analysis_outputs)

        print("\n" + "=" * 72)
        print("FULL PIPELINE COMPLETED")
        print("=" * 72)
        print(f"output_root: {output_root}")
        print(f"perturbation_output: {perturbation_output}")
        print(f"inference_output: {inference_output}")
        print(f"judge_output: {judge_output}")
        if analysis_outputs:
            for k, v in analysis_outputs.items():
                print(f"{k}: {v}")
        print("=" * 72)
        return 0
    except Exception as exc:
        logger.error("Full pipeline failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
