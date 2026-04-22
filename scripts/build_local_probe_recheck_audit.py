#!/usr/bin/env python3
"""Build a targeted recheck audit set from an existing local probe run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a targeted recheck audit set from probe_run.json.")
    parser.add_argument("--probe-run-file", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments/local_probe_qwen7b_recheck_audit")
    parser.add_argument("--controls-per-flip", type=int, default=1)
    return parser


def _load_samples(sample_file: str) -> dict[str, dict]:
    payload = json.loads(Path(sample_file).read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    return {str(row["sample_id"]): dict(row) for row in samples}


def _is_flip(row: dict) -> bool:
    return row.get("baseline_correct") is True and row.get("interference_correct") is False


def _is_matched_control_candidate(row: dict) -> bool:
    return row.get("baseline_correct") is True and row.get("interference_correct") is True


def _pick_controls(flip_rows: list[dict], control_rows: list[dict], controls_per_flip: int) -> list[dict]:
    used_ids: set[str] = set()
    selected: list[dict] = []
    for flip in flip_rows:
        flip_sample_type = str(flip.get("sample_type") or "")
        flip_condition_id = str(flip.get("condition_id") or "")

        exact = [
            row for row in control_rows
            if str(row.get("sample_id")) not in used_ids
            and str(row.get("sample_type") or "") == flip_sample_type
            and str(row.get("condition_id") or "") == flip_condition_id
        ]
        relaxed = [
            row for row in control_rows
            if str(row.get("sample_id")) not in used_ids
            and str(row.get("sample_type") or "") == flip_sample_type
        ]
        fallback = [row for row in control_rows if str(row.get("sample_id")) not in used_ids]
        pool = exact or relaxed or fallback
        for row in pool[: max(int(controls_per_flip), 0)]:
            used_ids.add(str(row.get("sample_id")))
            selected.append(
                {
                    **row,
                    "audit_group": "matched_nonflip_control",
                    "matched_to_sample_id": str(flip.get("sample_id")),
                }
            )
    return selected


def main() -> int:
    args = build_parser().parse_args()
    probe_path = Path(args.probe_run_file)
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    sample_map = _load_samples(str(payload["sample_file"]))
    comparisons = [dict(row) for row in payload.get("comparisons", [])]

    flip_rows = sorted(
        [row for row in comparisons if _is_flip(row)],
        key=lambda row: (str(row.get("sample_type") or ""), str(row.get("sample_id") or "")),
    )
    control_rows = sorted(
        [row for row in comparisons if _is_matched_control_candidate(row)],
        key=lambda row: (str(row.get("sample_type") or ""), str(row.get("sample_id") or "")),
    )
    matched_controls = _pick_controls(flip_rows, control_rows, int(args.controls_per_flip))

    audit_rows: list[dict] = []
    for row in flip_rows:
        audit_rows.append(
            {
                **sample_map.get(str(row["sample_id"]), {}),
                **row,
                "audit_group": "flip_all_in",
                "matched_to_sample_id": None,
            }
        )
    for row in matched_controls:
        audit_rows.append(
            {
                **sample_map.get(str(row["sample_id"]), {}),
                **row,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "probe_run_file": str(probe_path.resolve()),
        "sample_file": str(payload["sample_file"]),
        "num_flip_all_in": len(flip_rows),
        "num_matched_controls": len(matched_controls),
        "num_total_audit_rows": len(audit_rows),
        "controls_per_flip": int(args.controls_per_flip),
    }
    (output_dir / "recheck_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "recheck_audit_rows.json").write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "sample_id",
        "audit_group",
        "matched_to_sample_id",
        "sample_type",
        "condition_id",
        "baseline_predicted_answer",
        "interference_predicted_answer",
        "recheck_predicted_answer",
        "baseline_correct",
        "interference_correct",
        "recheck_correct",
        "interference_margin_delta",
        "recheck_margin_delta_vs_interference",
        "question_text",
        "ground_truth",
        "wrong_option",
        "prompt_prefix",
    ]
    with open(output_dir / "recheck_audit_rows.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow({key: row.get(key) for key in csv_fields})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
