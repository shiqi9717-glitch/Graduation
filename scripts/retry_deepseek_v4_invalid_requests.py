#!/usr/bin/env python3
"""Retry only invalid DeepSeek V4 pressure benchmark requests and merge outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_deepseek_v4_pressure_benchmark import (
    _build_records,
    _comparison_rows,
    _iter_requests,
    _parse_conditions,
    _resolve_model_config,
    _summary_rows,
    _usage_summary,
    _write_csv,
)
from src.inference.model_client import ModelClient
from src.open_model_probe.io_utils import prepare_output_dir, save_json, save_jsonl, sanitize_filename


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry invalid DeepSeek V4 benchmark requests only.")
    parser.add_argument("--source-run-dir", required=True, help="Existing run directory with invalid rows.")
    parser.add_argument("--output-root", required=True, help="Output root for merged retry run.")
    parser.add_argument("--models-config", default="config/models_config.json")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--input-price-per-1m", type=float, default=0.0)
    parser.add_argument("--output-price-per-1m", type=float, default=0.0)
    return parser


async def _amain(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_run_dir)
    manifest = json.loads((source_dir / "deepseek_v4_pressure_run.json").read_text(encoding="utf-8"))
    samples = _load_jsonl(source_dir / "benchmark_items.jsonl")
    old_response_rows = _load_jsonl(source_dir / "inference_responses.jsonl")
    old_parsed_rows = _load_jsonl(source_dir / "parsed_records.jsonl")
    invalid_ids = {str(row["request_id"]) for row in old_parsed_rows if row.get("is_invalid")}
    if not invalid_ids:
        raise ValueError(f"No invalid requests found in {source_dir}.")

    model = str(manifest["model"])
    conditions = _parse_conditions(",".join(manifest.get("conditions") or []))
    thinking = str(manifest.get("thinking") or "disabled")
    reasoning_effort = str(manifest.get("reasoning_effort") or "")
    all_requests = _iter_requests(
        samples,
        model,
        conditions,
        max_tokens=int(args.max_tokens),
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    retry_requests = [request for request in all_requests if request.request_id in invalid_ids]
    if len(retry_requests) != len(invalid_ids):
        found = {request.request_id for request in retry_requests}
        missing = sorted(invalid_ids - found)
        raise ValueError(f"Could not rebuild all invalid requests; missing={missing[:10]}")

    output_dir = prepare_output_dir(Path(args.output_root), run_name=sanitize_filename(model))
    save_jsonl(output_dir / "benchmark_items.jsonl", samples)

    model_config = _resolve_model_config(
        model,
        str(args.models_config),
        timeout=int(args.timeout),
        max_retries=int(args.max_retries),
        max_tokens=int(args.max_tokens),
    )
    async with ModelClient(model_config) as client:
        retry_responses = await client.batch_infer(retry_requests, concurrency_limit=int(args.concurrency))

    retry_response_rows, retry_parsed_rows = _build_records({str(row["task_id"]): row for row in samples}, retry_responses)
    retry_response_by_id = {str(row["request_id"]): row for row in retry_response_rows}
    retry_parsed_by_id = {str(row["request_id"]): row for row in retry_parsed_rows}

    merged_response_rows = [retry_response_by_id.get(str(row["request_id"]), row) for row in old_response_rows]
    merged_parsed_rows = [retry_parsed_by_id.get(str(row["request_id"]), row) for row in old_parsed_rows]
    comparisons = _comparison_rows(samples, merged_parsed_rows)
    summary = _summary_rows(merged_parsed_rows, comparisons, model)
    usage_summary = _usage_summary(
        merged_parsed_rows,
        input_price_per_1m=float(args.input_price_per_1m),
        output_price_per_1m=float(args.output_price_per_1m),
    )
    retry_usage_summary = _usage_summary(
        retry_parsed_rows,
        input_price_per_1m=float(args.input_price_per_1m),
        output_price_per_1m=float(args.output_price_per_1m),
    )

    save_jsonl(output_dir / "retry_inference_responses.jsonl", retry_response_rows)
    save_jsonl(output_dir / "retry_parsed_records.jsonl", retry_parsed_rows)
    save_jsonl(output_dir / "inference_responses.jsonl", merged_response_rows)
    save_jsonl(output_dir / "parsed_records.jsonl", merged_parsed_rows)
    save_jsonl(output_dir / "behavior_comparisons.jsonl", comparisons)
    _write_csv(output_dir / "behavior_summary.csv", summary)

    invalid_stats = {
        "model": model,
        "num_items": len(samples),
        "num_requests": len(merged_parsed_rows),
        "num_success": sum(1 for row in merged_parsed_rows if row["success"]),
        "num_invalid": sum(1 for row in merged_parsed_rows if row["is_invalid"]),
        "num_refusal_or_abstention": sum(1 for row in merged_parsed_rows if row["is_refusal_or_abstention"]),
        "num_retry_requests": len(retry_requests),
        "num_retry_invalid": sum(1 for row in retry_parsed_rows if row["is_invalid"]),
        "conditions": conditions,
        "usage": usage_summary,
        "retry_usage": retry_usage_summary,
    }
    save_json(output_dir / "invalid_refusal_stats.json", invalid_stats)
    save_json(
        output_dir / "deepseek_v4_pressure_run.json",
        {
            **manifest,
            "pipeline": "deepseek_v4_pressure_benchmark_invalid_retry_v1",
            "source_run_dir": str(source_dir.resolve()),
            "retry_invalid_only": True,
            "num_retry_requests": len(retry_requests),
            "max_tokens": int(args.max_tokens),
            "usage": usage_summary,
            "retry_usage": retry_usage_summary,
            "output_dir": str(output_dir.resolve()),
            "outputs": {
                "retry_inference_responses": str((output_dir / "retry_inference_responses.jsonl").resolve()),
                "retry_parsed_records": str((output_dir / "retry_parsed_records.jsonl").resolve()),
                "inference_responses": str((output_dir / "inference_responses.jsonl").resolve()),
                "parsed_records": str((output_dir / "parsed_records.jsonl").resolve()),
                "behavior_comparisons": str((output_dir / "behavior_comparisons.jsonl").resolve()),
                "behavior_summary": str((output_dir / "behavior_summary.csv").resolve()),
                "invalid_refusal_stats": str((output_dir / "invalid_refusal_stats.json").resolve()),
            },
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), **invalid_stats}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(_amain(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
