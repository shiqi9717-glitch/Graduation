"""Pair-level case report generator for original-vs-perturbed judge outputs."""

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SideRecord:
    pair_id: str
    side: str
    question: str
    answer: str
    score: Optional[float]
    reason: str


@dataclass
class PairRecord:
    pair_id: str
    original: SideRecord
    perturbed: SideRecord

    @property
    def delta(self) -> Optional[float]:
        if self.original.score is None or self.perturbed.score is None:
            return None
        return float(self.perturbed.score - self.original.score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pair-level sycophancy case reports.")
    parser.add_argument("--input-file", type=str, required=True, help="Input file path (JSONL or CSV).")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "jsonl", "csv"],
        help="Input file format.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="case_study_report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K cases for each direction.",
    )
    return parser.parse_args()


def _detect_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() == ".csv":
        return "csv"
    raise ValueError(f"Unsupported input file format: {path}")


def _safe_parse_obj(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _deep_get(data: Any, path: List[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_pair_id(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("pair_id"),
        _deep_get(row, ["metadata", "pair_id"]),
        _deep_get(row, ["metadata", "question_data", "metadata", "pair_id"]),
        _deep_get(row, ["question_data", "metadata", "pair_id"]),
    ]
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip()
    return None


def _extract_side(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("question_type"),
        _deep_get(row, ["metadata", "question_type"]),
        _deep_get(row, ["metadata", "question_data", "metadata", "question_type"]),
        _deep_get(row, ["question_data", "metadata", "question_type"]),
    ]
    for c in candidates:
        if c is None:
            continue
        side = str(c).strip().lower()
        if side in {"original", "perturbed"}:
            return side
    return None


def _extract_question(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("question"),
        row.get("question_text"),
        _deep_get(row, ["metadata", "question_data", "question_text"]),
        _deep_get(row, ["question_data", "question_text"]),
    ]
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip()
    return ""


def _extract_answer(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("answer"),
        row.get("response_text"),
        _deep_get(row, ["metadata", "response_text"]),
    ]
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip()
    return ""


def _extract_score(row: Dict[str, Any]) -> Optional[float]:
    candidates = [
        row.get("score"),
        _deep_get(row, ["raw_judgment", "score"]),
    ]
    for c in candidates:
        if c is None or str(c).strip() == "":
            continue
        try:
            return float(c)
        except Exception:
            continue
    return None


def _extract_reason(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("reason"),
        _deep_get(row, ["raw_judgment", "reason"]),
    ]
    for c in candidates:
        if c is not None and str(c).strip():
            return str(c).strip()
    return ""


def _normalize_row(row: Dict[str, Any]) -> Optional[SideRecord]:
    row = dict(row)
    if "metadata" in row:
        row["metadata"] = _safe_parse_obj(row["metadata"])
    if "question_data" in row:
        row["question_data"] = _safe_parse_obj(row["question_data"])
    if "raw_judgment" in row:
        row["raw_judgment"] = _safe_parse_obj(row["raw_judgment"])

    pair_id = _extract_pair_id(row)
    side = _extract_side(row)
    question = _extract_question(row)
    answer = _extract_answer(row)
    score = _extract_score(row)
    reason = _extract_reason(row)

    if not pair_id or not side:
        return None
    if not question or not answer:
        return None

    return SideRecord(
        pair_id=pair_id,
        side=side,
        question=question,
        answer=answer,
        score=score,
        reason=reason,
    )


def load_side_records(input_file: Path, input_format: str) -> List[SideRecord]:
    fmt = _detect_format(input_file, input_format)
    rows: List[Dict[str, Any]] = []

    if fmt == "jsonl":
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
    elif fmt == "csv":
        rows = pd.read_csv(input_file).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported input format: {fmt}")

    out: List[SideRecord] = []
    for row in rows:
        norm = _normalize_row(row)
        if norm is not None:
            out.append(norm)
    return out


def build_pairs(records: List[SideRecord]) -> List[PairRecord]:
    grouped: Dict[str, Dict[str, SideRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.pair_id, {})[rec.side] = rec

    pairs: List[PairRecord] = []
    for pair_id, sides in grouped.items():
        if "original" not in sides or "perturbed" not in sides:
            continue
        pairs.append(PairRecord(pair_id=pair_id, original=sides["original"], perturbed=sides["perturbed"]))
    return pairs


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _format_case(case: PairRecord) -> str:
    delta = case.delta
    delta_text = "NA" if delta is None else f"{delta:+.3f}"
    return (
        f"### Case ID: `{case.pair_id}`\n\n"
        f"**Case ID & 分数对比：** `{_fmt_score(case.original.score)}` vs "
        f"`{_fmt_score(case.perturbed.score)}` (Delta: `{delta_text}`)\n\n"
        f"**Q1 (原题):**\n\n{case.original.question}\n\n"
        f"**A1 (原回答):**\n\n{case.original.answer}\n\n"
        f"**Judge Reasoning 1:**\n\n{case.original.reason or 'N/A'}\n\n"
        f"**Q2 (扰动题):**\n\n{case.perturbed.question}\n\n"
        f"**A2 (扰动后回答):**\n\n{case.perturbed.answer}\n\n"
        f"**Judge Reasoning 2:**\n\n{case.perturbed.reason or 'N/A'}\n"
    )


def generate_report(pairs: List[PairRecord], output_file: Path, top_k: int) -> None:
    with_delta = [p for p in pairs if p.delta is not None]

    extreme_negative = sorted(
        [p for p in with_delta if p.delta < 0],
        key=lambda x: x.delta if x.delta is not None else 0.0,
    )[:top_k]

    positive_cases = sorted(
        [p for p in with_delta if p.delta > 0],
        key=lambda x: x.delta if x.delta is not None else 0.0,
        reverse=True,
    )[:top_k]

    lines: List[str] = []
    lines.append("# Sycophancy Case Study Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total complete pairs: {len(pairs)}")
    lines.append(f"- Pairs with valid delta: {len(with_delta)}")
    lines.append(f"- Extreme over-correction candidates (Delta < 0): {len(extreme_negative)}")
    lines.append(f"- Sycophancy-slide candidates (Delta > 0): {len(positive_cases)}")
    lines.append("")

    lines.append("## A. 极度反弹 / 过度纠偏 Top Cases")
    lines.append("")
    if not extreme_negative:
        lines.append("No cases found.")
    else:
        for case in extreme_negative:
            lines.append(_format_case(case))
            lines.append("")

    lines.append("## B. 阿谀奉承滑坡 / 成功诱导 Top Cases")
    lines.append("")
    if not positive_cases:
        lines.append("No cases found.")
    else:
        for case in positive_cases:
            lines.append(_format_case(case))
            lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    records = load_side_records(input_file=input_file, input_format=args.input_format)
    pairs = build_pairs(records)
    generate_report(pairs=pairs, output_file=output_file, top_k=args.top_k)

    print(f"Input records: {len(records)}")
    print(f"Complete pairs: {len(pairs)}")
    print(f"Report saved: {output_file}")


if __name__ == "__main__":
    main()
