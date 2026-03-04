"""Phase 4 statistics analyzer and report exporter."""

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.logging_config import logger


class StatsAnalyzer:
    """Analyze judge outputs and export publication-ready tables."""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

    @staticmethod
    def _detect_format(file_path: Path, input_format: str) -> str:
        if input_format != "auto":
            return input_format
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        if suffix == ".csv":
            return "csv"
        raise ValueError(f"Unsupported input format: {file_path}")

    @staticmethod
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

    @staticmethod
    def _deep_get(data: Any, path: List[str], default: Any = None) -> Any:
        cur = data
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _extract_model_name(self, row: Dict[str, Any]) -> str:
        candidates = [
            row.get("model_name"),
            self._deep_get(row, ["metadata", "model_name"]),
            self._deep_get(row, ["metadata", "question_data", "model_name"]),
            self._deep_get(row, ["metadata", "question_data", "metadata", "model_name"]),
            self._deep_get(row, ["metadata", "metadata", "model_name"]),
        ]
        for c in candidates:
            if c is not None and str(c).strip():
                return str(c).strip()
        return "unknown_model"

    def _extract_question_type(self, row: Dict[str, Any]) -> str:
        candidates = [
            self._deep_get(row, ["metadata", "question_type"]),
            self._deep_get(row, ["metadata", "question_data", "metadata", "question_type"]),
            self._deep_get(row, ["metadata", "metadata", "question_type"]),
            self._deep_get(row, ["metadata", "question_data", "question_type"]),
            row.get("question_type"),
        ]
        for c in candidates:
            if c is not None and str(c).strip():
                return str(c).strip()
        return "unknown"

    def _extract_pair_id(self, row: Dict[str, Any]) -> Optional[str]:
        candidates = [
            row.get("pair_id"),
            self._deep_get(row, ["metadata", "pair_id"]),
            self._deep_get(row, ["metadata", "question_data", "metadata", "pair_id"]),
            self._deep_get(row, ["metadata", "metadata", "pair_id"]),
            row.get("record_id"),
        ]
        for c in candidates:
            if c is not None and str(c).strip():
                return str(c).strip()
        return None

    def load(self, input_file: Path, input_format: str = "auto") -> pd.DataFrame:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Analysis input file not found: {input_file}")

        fmt = self._detect_format(input_file, input_format)
        rows: List[Dict[str, Any]] = []

        if fmt == "jsonl":
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    rows.append(json.loads(text))
            df = pd.DataFrame(rows)
        elif fmt == "csv":
            df = pd.read_csv(input_file)
        else:
            raise ValueError(f"Unsupported analysis format: {fmt}")

        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(self._safe_parse_obj)
        if "raw_judgment" in df.columns:
            df["raw_judgment"] = df["raw_judgment"].apply(self._safe_parse_obj)

        normalized_rows: List[Dict[str, Any]] = []
        for _, s in df.iterrows():
            row = s.to_dict()
            score = row.get("score")
            try:
                score = float(score) if score is not None and str(score) != "" else None
            except Exception:
                score = None
            success = row.get("success")
            if isinstance(success, str):
                success = success.lower() == "true"
            normalized_rows.append(
                {
                    "model_name": self._extract_model_name(row),
                    "question_type": self._extract_question_type(row),
                    "pair_id": self._extract_pair_id(row),
                    "score": score,
                    "success": bool(success),
                }
            )

        out = pd.DataFrame(normalized_rows)
        self.data = out
        logger.info("Loaded %s rows for analysis", len(out))
        return out

    def _compute_group_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            df.groupby(["model_name", "question_type"], dropna=False)["score"]
            .agg(["count", "mean", "var", "std"])
            .reset_index()
        )
        grouped = grouped.rename(
            columns={
                "count": "n",
                "mean": "score_mean",
                "var": "score_variance",
                "std": "score_std",
            }
        )
        return grouped

    def _compute_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        group_metrics = self._compute_group_metrics(df)
        all_types = sorted(group_metrics["question_type"].dropna().unique().tolist())

        rows: List[Dict[str, Any]] = []
        for model_name, mdf in group_metrics.groupby("model_name"):
            row: Dict[str, Any] = {"model_name": model_name}

            model_raw = df[df["model_name"] == model_name]
            row["overall_n"] = int(len(model_raw))
            row["overall_mean"] = float(model_raw["score"].mean())
            row["overall_variance"] = float(model_raw["score"].var()) if len(model_raw) > 1 else 0.0
            row["overall_std"] = float(model_raw["score"].std()) if len(model_raw) > 1 else 0.0

            for qt in all_types:
                q = mdf[mdf["question_type"] == qt]
                prefix = str(qt).lower().replace(" ", "_")
                if q.empty:
                    row[f"{prefix}_n"] = 0
                    row[f"{prefix}_mean"] = None
                    row[f"{prefix}_variance"] = None
                    row[f"{prefix}_std"] = None
                else:
                    row[f"{prefix}_n"] = int(q.iloc[0]["n"])
                    row[f"{prefix}_mean"] = float(q.iloc[0]["score_mean"])
                    row[f"{prefix}_variance"] = (
                        float(q.iloc[0]["score_variance"])
                        if pd.notna(q.iloc[0]["score_variance"])
                        else 0.0
                    )
                    row[f"{prefix}_std"] = (
                        float(q.iloc[0]["score_std"])
                        if pd.notna(q.iloc[0]["score_std"])
                        else 0.0
                    )

            rows.append(row)

        return pd.DataFrame(rows)

    def _compute_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        # Requires pair_id + original/perturbed question_type.
        needed_types = {"original", "perturbed"}
        delta_rows: List[Dict[str, Any]] = []

        for model_name, mdf in df.groupby("model_name"):
            paired = mdf.dropna(subset=["pair_id"]).copy()
            if paired.empty:
                continue

            per_pair: List[float] = []
            for pair_id, pdf in paired.groupby("pair_id"):
                types = set(str(t).lower() for t in pdf["question_type"].tolist())
                if not needed_types.issubset(types):
                    continue

                orig = pdf[pdf["question_type"].str.lower() == "original"]["score"].mean()
                pert = pdf[pdf["question_type"].str.lower() == "perturbed"]["score"].mean()
                if pd.notna(orig) and pd.notna(pert):
                    per_pair.append(abs(float(pert) - float(orig)))

            if per_pair:
                delta_rows.append(
                    {
                        "model_name": model_name,
                        "paired_samples": len(per_pair),
                        "delta_abs_mean": float(pd.Series(per_pair).mean()),
                        "delta_abs_variance": float(pd.Series(per_pair).var()) if len(per_pair) > 1 else 0.0,
                        "delta_abs_std": float(pd.Series(per_pair).std()) if len(per_pair) > 1 else 0.0,
                    }
                )

        return pd.DataFrame(delta_rows)

    def analyze(self) -> Dict[str, pd.DataFrame]:
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        valid = self.data[(self.data["success"] == True) & (self.data["score"].notna())]  # noqa: E712
        if valid.empty:
            raise ValueError("No valid scored records found for analysis.")

        group_metrics = self._compute_group_metrics(valid)
        model_summary = self._compute_model_summary(valid)
        delta_df = self._compute_delta(valid)

        if not delta_df.empty:
            model_summary = model_summary.merge(delta_df, on="model_name", how="left")

        return {
            "valid_records": valid,
            "group_metrics": group_metrics,
            "model_summary": model_summary,
            "delta_metrics": delta_df,
        }

    def export(self, analysis: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        paths: Dict[str, Path] = {}

        final_report = output_dir / f"final_report_{timestamp}.csv"
        analysis["model_summary"].to_csv(final_report, index=False, encoding="utf-8-sig")
        paths["final_report"] = final_report

        grouped_report = output_dir / f"group_metrics_{timestamp}.csv"
        analysis["group_metrics"].to_csv(grouped_report, index=False, encoding="utf-8-sig")
        paths["group_metrics"] = grouped_report

        valid_records = output_dir / f"valid_scored_records_{timestamp}.csv"
        analysis["valid_records"].to_csv(valid_records, index=False, encoding="utf-8-sig")
        paths["valid_records"] = valid_records

        if not analysis["delta_metrics"].empty:
            delta_report = output_dir / f"delta_metrics_{timestamp}.csv"
            analysis["delta_metrics"].to_csv(delta_report, index=False, encoding="utf-8-sig")
            paths["delta_metrics"] = delta_report

        summary_json = output_dir / f"analysis_summary_{timestamp}.json"
        summary_payload = {
            "generated_at": datetime.now().isoformat(),
            "rows_total": int(len(self.data)) if self.data is not None else 0,
            "rows_valid": int(len(analysis["valid_records"])),
            "models": sorted(analysis["model_summary"]["model_name"].tolist()),
            "files": {k: str(v) for k, v in paths.items()},
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        paths["summary_json"] = summary_json

        return paths

    def run(
        self,
        input_file: Path,
        input_format: str = "auto",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        self.load(input_file=input_file, input_format=input_format)
        analysis = self.analyze()
        if output_dir is None:
            output_dir = Path("data/results/analysis")
        return self.export(analysis, output_dir)
