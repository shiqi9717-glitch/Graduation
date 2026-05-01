#!/usr/bin/env python3
"""Summarize a human-labeled audit CSV with graceful handling of missing labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_COLUMNS = [
    "human_stance_drift",
    "human_pressured_compliance",
    "human_recovery",
    "human_baseline_damage",
    "human_over_conservatism_or_refusal",
    "human_semantic_quality",
    "human_rationale_reasonable",
]
PROXY_ALIGNMENT = {
    "human_stance_drift": "stance_drift_proxy",
    "human_pressured_compliance": "pressured_compliance_proxy",
    "human_recovery": "recovery_proxy",
    "human_baseline_damage": "baseline_damage_proxy",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text


def _label_present(row: dict[str, Any], column: str) -> bool:
    return _normalize_label(row.get(column)) not in {"", "NA"}


def _bool_from_label(value: Any) -> bool | None:
    text = _normalize_label(value)
    if text in {"", "NA"}:
        return None
    try:
        return float(text) >= 1.0
    except Exception:
        return None


def _bool_from_proxy(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a human-audit CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(input_csv)
    fieldnames = list(rows[0].keys()) if rows else []
    missing_columns = [column for column in LABEL_COLUMNS if column not in fieldnames]

    if missing_columns:
        payload = {
            "status": "missing_label_columns",
            "input_csv": str(input_csv),
            "missing_columns": missing_columns,
            "num_rows": len(rows),
        }
        (output_dir / "audit_summary.md").write_text(
            "\n".join(
                [
                    "# Audit Summary",
                    "",
                    "No aggregate label summary could be computed because the following expected label columns are missing:",
                    *[f"- `{column}`" for column in missing_columns],
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "audit_summary.csv").write_text("status,missing_columns,num_rows\nmissing_label_columns,\"" + ";".join(missing_columns) + f"\",{len(rows)}\n", encoding="utf-8")
        (output_dir / "audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    distributions: dict[str, dict[str, int]] = {}
    completed_counts: dict[str, int] = {}
    for column in LABEL_COLUMNS:
        counter = Counter(_normalize_label(row.get(column)) for row in rows if _normalize_label(row.get(column)))
        distributions[column] = dict(counter)
        completed_counts[column] = sum(counter.values())

    group_rows: list[dict[str, Any]] = []
    for group_field in ("pressure_type", "setting"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(group_field) or "")].append(row)
        for group_value, subset in sorted(grouped.items()):
            out = {"group_field": group_field, "group_value": group_value, "num_rows": len(subset)}
            for column in LABEL_COLUMNS:
                labeled = [_normalize_label(row.get(column)) for row in subset if _normalize_label(row.get(column))]
                out[f"{column}_count"] = len(labeled)
                for label_value in ("0", "1", "2", "NA"):
                    out[f"{column}_{label_value}"] = sum(1 for value in labeled if value == label_value)
            group_rows.append(out)

    agreement_rows: list[dict[str, Any]] = []
    for human_col, proxy_col in PROXY_ALIGNMENT.items():
        comparable = []
        for row in rows:
            human = _bool_from_label(row.get(human_col))
            proxy = _bool_from_proxy(row.get(proxy_col))
            if human is None or proxy is None:
                continue
            comparable.append((human, proxy, row))
        accuracy = (sum(1 for human, proxy, _ in comparable if human == proxy) / len(comparable)) if comparable else None
        agreement_rows.append(
            {
                "human_column": human_col,
                "proxy_column": proxy_col,
                "num_comparable": len(comparable),
                "agreement_rate": accuracy,
            }
        )

    payload = {
        "status": "ok" if any(count > 0 for count in completed_counts.values()) else "no_completed_labels",
        "input_csv": str(input_csv),
        "num_rows": len(rows),
        "completed_counts": completed_counts,
        "label_distributions": distributions,
        "proxy_agreement": agreement_rows,
    }
    _write_csv(output_dir / "audit_summary.csv", group_rows if group_rows else [{"group_field": "", "group_value": "", "num_rows": 0}])
    (output_dir / "audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "audit_summary.md").write_text(
        "\n".join(
            [
                "# Audit Summary",
                "",
                f"- Input CSV: `{input_csv}`",
                f"- Rows: `{len(rows)}`",
                f"- Status: `{payload['status']}`",
                "",
                "## Completed Labels",
                *[f"- `{column}`: `{completed_counts[column]}`" for column in LABEL_COLUMNS],
                "",
                "## Proxy Agreement",
                *[
                    f"- `{row['human_column']}` vs `{row['proxy_column']}`: comparable=`{row['num_comparable']}`, agreement=`{row['agreement_rate']}`"
                    for row in agreement_rows
                ],
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "status": payload["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
