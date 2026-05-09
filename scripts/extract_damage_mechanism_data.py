#!/usr/bin/env python3
"""Extract margin and option-ranking damage signals from existing belief-causal artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.open_model_probe.io_utils import prepare_output_dir, save_json


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _ranked_options(answer_logits: Dict[str, Any]) -> List[str]:
    return [item[0] for item in sorted(((str(k), float(v)) for k, v in dict(answer_logits).items()), key=lambda item: item[1], reverse=True)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract damage mechanism data from existing belief-causal runs.")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", default="outputs/experiments/damage_mechanism_analysis")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    records = _read_jsonl(run_dir / "belief_causal_records.jsonl")
    comparisons = _read_jsonl(run_dir / "belief_causal_comparisons.jsonl")
    projection_rows = _read_csv(run_dir / "projection_alignment_diagnostic.csv")

    output_dir = prepare_output_dir(Path(args.output_root), run_name=str(args.model_label))
    comparison_by_id = {(str(row["item_id"]), str(row["method"])): dict(row) for row in comparisons}
    projection_by_id = {str(row["item"]): dict(row) for row in projection_rows}

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in records:
        grouped[(str(row["item_id"]), str(row["method"]))] = grouped.get((str(row["item_id"]), str(row["method"])), {})
        grouped[(str(row["item_id"]), str(row["method"]))][str(row["scenario"])] = dict(row)

    item_rows: List[Dict[str, Any]] = []
    taxonomy_rows: List[Dict[str, Any]] = []
    for item_id, method in sorted(grouped):
        if method == "no_intervention":
            continue
        before = grouped.get((item_id, "no_intervention"), {})
        after = grouped.get((item_id, method), {})
        if "baseline" not in before or "baseline" not in after:
            continue
        baseline_before = before["baseline"]
        baseline_after = after["baseline"]
        ranked_before = _ranked_options(baseline_before.get("answer_logits", {}))
        ranked_after = _ranked_options(baseline_after.get("answer_logits", {}))
        comp = comparison_by_id.get((item_id, method), {})
        projection = projection_by_id.get(item_id, {})
        answer_matching = str(comp.get("answer_matching_behavior") or "")
        answer_nonmatching = str(comp.get("answer_not_matching_behavior") or "")
        before_rank = ranked_before.index(answer_matching) + 1 if answer_matching in ranked_before else ""
        after_rank = ranked_after.index(answer_matching) + 1 if answer_matching in ranked_after else ""
        flip = bool(comp.get("baseline_damage"))
        margin_before = baseline_before.get("correct_wrong_margin")
        margin_after = baseline_after.get("correct_wrong_margin")
        margin_delta = (float(margin_after) - float(margin_before)) if margin_before is not None and margin_after is not None else ""
        top1_before = ranked_before[0] if ranked_before else ""
        top1_after = ranked_after[0] if ranked_after else ""
        if flip:
            if top1_after and top1_after == answer_nonmatching:
                damage_type = "answer_flip"
            elif margin_delta != "" and float(margin_delta) < 0:
                damage_type = "margin_collapse"
            else:
                damage_type = "unstable_option_ranking"
        else:
            if margin_delta != "" and float(margin_delta) < 0:
                damage_type = "margin_collapse_without_flip"
            else:
                damage_type = "no_damage"
        if top1_after and top1_after not in {"", answer_matching, answer_nonmatching}:
            damage_type = "non_target_overcorrection"

        row = {
            "item_id": item_id,
            "method": method,
            "baseline_damage": bool(comp.get("baseline_damage")),
            "baseline_choice_before": baseline_before.get("parsed_answer", ""),
            "baseline_choice_after": baseline_after.get("parsed_answer", ""),
            "baseline_margin_before": margin_before,
            "baseline_margin_after": margin_after,
            "baseline_margin_delta": margin_delta,
            "correct_option_rank_before": before_rank,
            "correct_option_rank_after": after_rank,
            "top1_before": top1_before,
            "top1_after": top1_after,
            "belief_logit_delta_vs_no": projection.get("belief_logit_delta_vs_no", ""),
            "negative_logit_delta_vs_no": projection.get("negative_logit_delta_vs_no", ""),
            "damage_type": damage_type,
        }
        item_rows.append(row)
        if bool(comp.get("baseline_damage")):
            taxonomy_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    methods = sorted({row["method"] for row in item_rows})
    for method in methods:
        subset = [row for row in item_rows if row["method"] == method]
        damage_subset = [row for row in subset if row["baseline_damage"]]
        summary_rows.append(
            {
                "model": str(args.model_label),
                "method": method,
                "num_items": len(subset),
                "damage_rate": (len(damage_subset) / len(subset)) if subset else "",
                "mean_belief_logit_delta_vs_no": (
                    sum(float(row["belief_logit_delta_vs_no"]) for row in subset if row["belief_logit_delta_vs_no"] != "") / len([row for row in subset if row["belief_logit_delta_vs_no"] != ""])
                ) if any(row["belief_logit_delta_vs_no"] != "" for row in subset) else "",
                "mean_negative_logit_delta_vs_no": (
                    sum(float(row["negative_logit_delta_vs_no"]) for row in subset if row["negative_logit_delta_vs_no"] != "") / len([row for row in subset if row["negative_logit_delta_vs_no"] != ""])
                ) if any(row["negative_logit_delta_vs_no"] != "" for row in subset) else "",
                "mean_baseline_margin_delta_on_damage": (
                    sum(float(row["baseline_margin_delta"]) for row in damage_subset if row["baseline_margin_delta"] != "") / len([row for row in damage_subset if row["baseline_margin_delta"] != ""])
                ) if any(row["baseline_margin_delta"] != "" for row in damage_subset) else "",
                "answer_flip_count": sum(1 for row in damage_subset if row["damage_type"] == "answer_flip"),
                "non_target_overcorrection_count": sum(1 for row in damage_subset if row["damage_type"] == "non_target_overcorrection"),
                "unstable_option_ranking_count": sum(1 for row in damage_subset if row["damage_type"] == "unstable_option_ranking"),
                "margin_collapse_count": sum(1 for row in damage_subset if row["damage_type"] == "margin_collapse"),
            }
        )

    _write_csv(output_dir / "damage_item_analysis.csv", item_rows)
    _write_csv(output_dir / "damage_taxonomy.csv", taxonomy_rows)
    _write_csv(output_dir / "damage_signal_summary.csv", summary_rows)
    save_json(
        output_dir / "damage_mechanism_manifest.json",
        {
            "pipeline": "damage_mechanism_extraction_v1",
            "model_label": str(args.model_label),
            "run_dir": str(run_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "num_item_rows": len(item_rows),
            "num_damage_rows": len(taxonomy_rows),
        },
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), "num_item_rows": len(item_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
