#!/usr/bin/env python3
"""Repair historical guarded recheck outputs using the current merge logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_deepseek_guarded_recheck import (
    DEEPSEEK_REASONER_SKIP_REASON,
    REASONER_MODEL_NAME,
    _assign_manifest_alignment,
    _build_group_rows,
    _build_summary,
    _dedupe_scored_rows,
    _ensure_text_series,
    _merge_results,
    _normalize_scored_dataset,
    _resolve_change_gate,
    _resolve_trigger_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair historical guarded output directories in place.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments",
        help="Root directory to scan for guarded output folders.",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Optional specific output directory to repair. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report planned repairs without writing files.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _detect_recheck_mode(summary: Dict[str, Any], manifest: pd.DataFrame, judge_results: List[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    recheck_mode = str(summary.get("recheck_mode") or "").strip()
    override = str(summary.get("recheck_model_override") or summary.get("judge_model_name") or "").strip()
    if recheck_mode in {"same_model", "override_model"}:
        return recheck_mode, (override or None)

    if override:
        return "override_model", override

    if judge_results:
        judge_df = pd.DataFrame(judge_results)
        if "recheck_model_name" in judge_df.columns:
            used_models = sorted({str(v).strip() for v in judge_df["recheck_model_name"].fillna("").astype(str) if str(v).strip()})
            manifest_models = sorted(
                {
                    str(v).strip()
                    for v in _ensure_text_series(_normalize_scored_dataset(manifest), "model_name").tolist()
                    if str(v).strip()
                }
            )
            if len(used_models) == 1 and used_models != manifest_models:
                return "override_model", used_models[0]

    return "same_model", None


def _infer_prompt_style(summary: Dict[str, Any], directory: Path) -> tuple[str, str]:
    prompt_style = str(summary.get("prompt_style") or "").strip()
    reasoner_prompt_style = str(summary.get("reasoner_prompt_style") or "").strip()
    dirname = directory.as_posix()

    if not prompt_style:
        prompt_style = "standard"
    if not reasoner_prompt_style:
        if "reasoner_minimal" in dirname:
            reasoner_prompt_style = "reasoner_minimal"
        elif "reasoner_short" in dirname:
            reasoner_prompt_style = "reasoner_short"
        else:
            reasoner_prompt_style = "standard"
    return prompt_style, reasoner_prompt_style


def _apply_reasoner_skip_reason(manifest: pd.DataFrame, recheck_mode: str, recheck_model_override: Optional[str]) -> pd.DataFrame:
    out = manifest.copy()
    if "recheck_skip_reason" not in out.columns:
        out["recheck_skip_reason"] = ""
    else:
        out["recheck_skip_reason"] = _ensure_text_series(out, "recheck_skip_reason")

    if recheck_mode == "same_model" and not recheck_model_override:
        reasoner_skip_mask = (
            out["triggered"].astype(int).eq(1)
            & _ensure_text_series(out, "model_name").str.strip().str.lower().eq(REASONER_MODEL_NAME)
        )
        out.loc[reasoner_skip_mask, "recheck_skip_reason"] = DEEPSEEK_REASONER_SKIP_REASON
    return out


def _infer_policy_and_gate(summary: Dict[str, Any]) -> tuple[Any, Any]:
    trigger_policy = summary.get("trigger_policy") or {}
    change_gate = summary.get("change_gate") or {}
    threshold = float(summary.get("default_threshold", summary.get("threshold", 0.55)) or 0.55)
    reasoner_threshold = float(summary.get("reasoner_threshold", 0.70) or 0.70)
    policy_name = str(trigger_policy.get("policy_name") or "global")
    gate_name = str(change_gate.get("gate_name") or "none")
    return (
        _resolve_trigger_policy(policy_name, threshold, reasoner_threshold),
        _resolve_change_gate(gate_name),
    )


def _repair_directory(directory: Path, dry_run: bool) -> Optional[Dict[str, Any]]:
    manifest_path = directory / "sample_manifest.csv"
    judge_path = directory / "judge_recheck_results.jsonl"
    guarded_path = directory / "guarded_samples.csv"
    summary_path = directory / "guarded_eval_summary.json"
    grouped_path = directory / "guarded_eval_by_group.csv"

    if not manifest_path.exists() or not guarded_path.exists():
        return None

    manifest_original = pd.read_csv(manifest_path)
    judge_results = _load_jsonl(judge_path)
    summary = _load_json(summary_path)

    manifest_fixed = _assign_manifest_alignment(_dedupe_scored_rows(_normalize_scored_dataset(manifest_original)))
    policy_config, change_gate_config = _infer_policy_and_gate(summary)
    recheck_mode, recheck_model_override = _detect_recheck_mode(summary, manifest_fixed, judge_results)
    prompt_style, reasoner_prompt_style = _infer_prompt_style(summary, directory)
    manifest_fixed = _apply_reasoner_skip_reason(manifest_fixed, recheck_mode, recheck_model_override)

    final_df = _merge_results(
        manifest_fixed,
        judge_results,
        trigger_policy_config=policy_config,
        change_gate_config=change_gate_config,
    )
    grouped = _build_group_rows(final_df)

    detector_model_path = str(summary.get("detector_model_path") or "")
    source_num_rows = int(summary.get("source_num_rows") or len(manifest_original))
    filtered_num_rows = int(summary.get("filtered_num_rows") or len(manifest_fixed))
    model_filters = summary.get("model_filters") or []
    deepseek_only = bool(summary.get("deepseek_only", False))
    threshold_name = str(summary.get("threshold_name") or "matched_trigger_budget")

    repaired_summary = _build_summary(
        final_df=final_df,
        threshold_name=threshold_name,
        policy_config=policy_config,
        change_gate_config=change_gate_config,
        detector_model_path=detector_model_path,
        recheck_mode=recheck_mode,
        recheck_model_override=recheck_model_override,
        source_num_rows=source_num_rows,
        filtered_num_rows=filtered_num_rows,
        model_filters=list(model_filters),
        deepseek_only=deepseek_only,
        prompt_style=prompt_style,
        reasoner_prompt_style=reasoner_prompt_style,
    )
    repaired_summary["sample_manifest"] = str(manifest_path.resolve())
    if judge_path.exists():
        repaired_summary["judge_recheck_results"] = str(judge_path.resolve())
    repaired_summary["guarded_samples"] = str(guarded_path.resolve())
    repaired_summary["guarded_eval_by_group"] = str(grouped_path.resolve())

    before_guarded_rows = None
    if guarded_path.exists():
        before_guarded_rows = len(pd.read_csv(guarded_path))

    result = {
        "directory": str(directory),
        "manifest_rows_before": int(len(manifest_original)),
        "manifest_rows_after": int(len(manifest_fixed)),
        "guarded_rows_before": int(before_guarded_rows or 0),
        "guarded_rows_after": int(len(final_df)),
        "had_manifest_row_id": int("manifest_row_id" in manifest_original.columns),
        "had_occurrence_index": int("occurrence_index" in manifest_original.columns),
    }

    if dry_run:
        return result

    manifest_fixed.to_csv(manifest_path, index=False)
    final_df.to_csv(guarded_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(repaired_summary, f, ensure_ascii=False, indent=2)
    return result


def _iter_candidate_directories(root: Path, explicit_dirs: Iterable[str]) -> List[Path]:
    if explicit_dirs:
        return [Path(item).expanduser().resolve() for item in explicit_dirs]
    candidates = []
    for manifest_path in sorted(root.rglob("sample_manifest.csv")):
        directory = manifest_path.parent
        if (directory / "guarded_samples.csv").exists():
            candidates.append(directory)
    return candidates


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    candidates = _iter_candidate_directories(root, args.dir)

    repaired: List[Dict[str, Any]] = []
    for directory in candidates:
        result = _repair_directory(directory, dry_run=bool(args.dry_run))
        if result is not None:
            repaired.append(result)

    print(json.dumps({"count": len(repaired), "directories": repaired}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
