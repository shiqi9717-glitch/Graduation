"""Phase 4 visualization module for publication-ready figures."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.logging_config import logger


class StatsVisualizer:
    """Generate delta bar chart and score distribution boxplot from Phase 4 CSV files."""

    def __init__(self, dpi: int = 300):
        if dpi <= 0:
            raise ValueError("dpi must be > 0")
        self.dpi = dpi
        sns.set_theme(style="whitegrid", context="talk")
        plt.style.use("seaborn-v0_8-whitegrid")

    @staticmethod
    def _load_csv(file_path: Path) -> pd.DataFrame:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")
        return df

    @staticmethod
    def _normalize_question_type(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower()

    def _as_valid_records(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"model_name", "question_type", "score"}
        if not required.issubset(df.columns):
            return None

        out = df.copy()
        out["model_name"] = out["model_name"].astype(str).str.strip()
        out["question_type"] = self._normalize_question_type(out["question_type"])
        out["score"] = pd.to_numeric(out["score"], errors="coerce")

        if "success" in out.columns:
            success = out["success"]
            if success.dtype == object:
                success = success.astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False}
                )
            out = out[success.fillna(False) == True]  # noqa: E712

        out = out.dropna(subset=["model_name", "question_type", "score"])
        if out.empty:
            return None
        return out

    def _as_group_metrics(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"model_name", "question_type", "score_mean"}
        if not required.issubset(df.columns):
            return None

        out = df.copy()
        out["model_name"] = out["model_name"].astype(str).str.strip()
        out["question_type"] = self._normalize_question_type(out["question_type"])
        out["score_mean"] = pd.to_numeric(out["score_mean"], errors="coerce")
        out = out.dropna(subset=["model_name", "question_type", "score_mean"])
        if out.empty:
            return None
        return out

    @staticmethod
    def _as_cross_model_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"model_name", "baseline_accuracy", "treated_accuracy_mean", "treated_wrong_option_follow_rate_mean"}
        if not required.issubset(df.columns):
            return None
        out = df.copy()
        for col in ("baseline_accuracy", "treated_accuracy_mean", "treated_wrong_option_follow_rate_mean"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "reasoning_mode" not in out.columns:
            out["reasoning_mode"] = "unknown"
        if "model_family" not in out.columns:
            out["model_family"] = "unknown"
        return out.dropna(subset=["model_name"]).reset_index(drop=True)

    @staticmethod
    def _as_cross_model_factor_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {
            "model_name",
            "authority_level",
            "confidence_level",
            "explicit_wrong_option",
            "accuracy",
            "wrong_option_follow_rate",
        }
        if not required.issubset(df.columns):
            return None
        out = df.copy()
        for col in ("authority_level", "confidence_level", "explicit_wrong_option", "accuracy", "wrong_option_follow_rate", "sycophancy_rate"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.dropna(subset=["model_name"]).reset_index(drop=True)

    @staticmethod
    def _as_arm_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"accuracy", "sycophancy_rate"}
        if not required.issubset(df.columns):
            return None

        out = df.copy()
        if "model_name" not in out.columns:
            out["model_name"] = "overall"
        if "condition_id" not in out.columns:
            if "question_type" not in out.columns:
                return None
            out["condition_id"] = out["question_type"]
        if "condition_label" not in out.columns:
            out["condition_label"] = out["condition_id"]
        if "question_type" not in out.columns:
            out["question_type"] = out["condition_id"]
        out["model_name"] = out["model_name"].astype(str).str.strip()
        out["condition_id"] = out["condition_id"].astype(str).str.strip().str.lower()
        out["condition_label"] = out["condition_label"].astype(str).str.strip()
        out["question_type"] = out["question_type"].astype(str).str.strip().str.lower()
        if "authority_level" not in out.columns:
            out["authority_level"] = out.get("pressure_level")
        if "confidence_level" not in out.columns:
            out["confidence_level"] = np.nan
        if "explicit_wrong_option" not in out.columns:
            out["explicit_wrong_option"] = 0
        if "is_control" not in out.columns:
            out["is_control"] = False
        out["authority_level"] = pd.to_numeric(out["authority_level"], errors="coerce")
        out["confidence_level"] = pd.to_numeric(out["confidence_level"], errors="coerce")
        out["explicit_wrong_option"] = pd.to_numeric(out["explicit_wrong_option"], errors="coerce").fillna(0).astype(int)
        out["is_control"] = out["is_control"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        out["pressure_level"] = out["authority_level"]
        out["accuracy"] = pd.to_numeric(out["accuracy"], errors="coerce")
        out["sycophancy_rate"] = pd.to_numeric(out["sycophancy_rate"], errors="coerce")
        out["is_placebo"] = out["is_control"] & out["explicit_wrong_option"].eq(1)
        if "arm_label" not in out.columns:
            out["arm_label"] = out["condition_label"]
        out = out.dropna(subset=["condition_id", "accuracy", "sycophancy_rate"])
        if out.empty:
            return None
        return out.sort_values(["model_name", "is_control", "authority_level", "confidence_level", "explicit_wrong_option", "condition_id"]).reset_index(drop=True)

    @staticmethod
    def _as_category_arm_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        required = {"category", "sycophancy_rate"}
        if not required.issubset(df.columns):
            return None

        out = df.copy()
        if "model_name" not in out.columns:
            out["model_name"] = "overall"
        if "condition_id" not in out.columns:
            if "question_type" not in out.columns:
                return None
            out["condition_id"] = out["question_type"]
        if "condition_label" not in out.columns:
            out["condition_label"] = out["condition_id"]
        if "question_type" not in out.columns:
            out["question_type"] = out["condition_id"]
        out["model_name"] = out["model_name"].astype(str).str.strip()
        out["category"] = out["category"].astype(str).str.strip()
        out["condition_id"] = out["condition_id"].astype(str).str.strip().str.lower()
        out["condition_label"] = out["condition_label"].astype(str).str.strip()
        out["question_type"] = out["question_type"].astype(str).str.strip().str.lower()
        if "authority_level" not in out.columns:
            out["authority_level"] = out.get("pressure_level")
        if "confidence_level" not in out.columns:
            out["confidence_level"] = np.nan
        if "explicit_wrong_option" not in out.columns:
            out["explicit_wrong_option"] = 0
        if "is_control" not in out.columns:
            out["is_control"] = False
        out["authority_level"] = pd.to_numeric(out["authority_level"], errors="coerce")
        out["confidence_level"] = pd.to_numeric(out["confidence_level"], errors="coerce")
        out["explicit_wrong_option"] = pd.to_numeric(out["explicit_wrong_option"], errors="coerce").fillna(0).astype(int)
        out["is_control"] = out["is_control"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        out["pressure_level"] = out["authority_level"]
        out["sycophancy_rate"] = pd.to_numeric(out["sycophancy_rate"], errors="coerce")
        out["is_placebo"] = out["is_control"] & out["explicit_wrong_option"].eq(1)
        for col in (
            "accuracy",
            "accuracy_sem",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "sycophancy_sem",
            "sycophancy_ci95_low",
            "sycophancy_ci95_high",
        ):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["category", "condition_id", "sycophancy_rate"])
        if out.empty:
            return None
        return out.sort_values(["model_name", "category", "is_control", "authority_level", "confidence_level", "explicit_wrong_option", "condition_id"]).reset_index(drop=True)

    @staticmethod
    def _save_figure(
        fig: plt.Figure,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
        dpi: int,
    ) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, Path] = {}

        for ext in formats:
            ext_clean = ext.strip().lower()
            if ext_clean not in {"png", "pdf"}:
                raise ValueError(f"Unsupported output format: {ext_clean}")
            path = output_dir / f"{stem}.{ext_clean}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved[ext_clean] = path
        plt.close(fig)
        return saved

    def _build_delta_table(
        self,
        valid_records: pd.DataFrame,
        group_metrics: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if group_metrics is not None:
            base = group_metrics.rename(columns={"score_mean": "score"}).copy()
            base = base[["model_name", "question_type", "score"]]
        else:
            base = (
                valid_records.groupby(["model_name", "question_type"], as_index=False)["score"]
                .mean()
                .copy()
            )

        base = base[base["question_type"].isin(["original", "perturbed"])]
        if base.empty:
            raise ValueError("No original/perturbed records found for delta chart.")

        pivot = base.pivot_table(
            index="model_name",
            columns="question_type",
            values="score",
            aggfunc="mean",
        ).reset_index()
        if "original" not in pivot.columns or "perturbed" not in pivot.columns:
            raise ValueError("Delta chart requires both original and perturbed question_type.")

        pivot["delta"] = pivot["perturbed"] - pivot["original"]
        pivot = pivot.sort_values("delta", ascending=False).reset_index(drop=True)
        return pivot

    def plot_delta_bar_chart(
        self,
        delta_table: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        models = delta_table["model_name"].tolist()
        orig = delta_table["original"].tolist()
        pert = delta_table["perturbed"].tolist()

        x = list(range(len(models)))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 6))
        bars_orig = ax.bar(
            [i - width / 2 for i in x],
            orig,
            width=width,
            label="Original",
            color="#4C78A8",
        )
        bars_pert = ax.bar(
            [i + width / 2 for i in x],
            pert,
            width=width,
            label="Perturbed",
            color="#F58518",
        )

        for i, row in delta_table.iterrows():
            y = max(row["original"], row["perturbed"]) + 0.05
            ax.text(i, y, f"Δ={row['delta']:+.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title("Score Comparison Before/After Perturbation", fontsize=14, pad=12)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Mean Judge Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=True)
        ax.bar_label(bars_orig, fmt="%.2f", padding=2, fontsize=8)
        ax.bar_label(bars_pert, fmt="%.2f", padding=2, fontsize=8)

        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_score_boxplot(
        self,
        valid_records: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = valid_records[valid_records["question_type"].isin(["original", "perturbed"])].copy()
        if df.empty:
            raise ValueError("No original/perturbed score records found for boxplot.")

        models = sorted(df["model_name"].unique().tolist())
        question_types = ["original", "perturbed"]

        data: List[List[float]] = []
        positions: List[float] = []
        colors: List[str] = []
        tick_positions: List[float] = []
        tick_labels: List[str] = []

        color_map = {"original": "#4C78A8", "perturbed": "#F58518"}
        pos = 1.0

        for model in models:
            model_positions: List[float] = []
            for qtype in question_types:
                values = (
                    df[(df["model_name"] == model) & (df["question_type"] == qtype)]["score"]
                    .astype(float)
                    .tolist()
                )
                if not values:
                    continue
                data.append(values)
                positions.append(pos)
                colors.append(color_map[qtype])
                model_positions.append(pos)
                pos += 1.0
            if model_positions:
                tick_positions.append(sum(model_positions) / len(model_positions))
                tick_labels.append(model)
                pos += 0.8

        if not data:
            raise ValueError("No valid score data for boxplot.")

        fig, ax = plt.subplots(figsize=(13, 6))
        box = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=0.65,
            medianprops={"color": "#1f1f1f", "linewidth": 1.4},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title("Score Distribution Across Repeated Samples", fontsize=14, pad=12)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Judge Score", fontsize=11)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_ylim(bottom=0)
        ax.legend(
            handles=[
                Patch(facecolor=color_map["original"], edgecolor="black", alpha=0.75, label="Original"),
                Patch(
                    facecolor=color_map["perturbed"],
                    edgecolor="black",
                    alpha=0.75,
                    label="Perturbed",
                ),
            ],
            frameon=True,
            loc="upper right",
        )

        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_control_vs_treated_overview(
        self,
        arm_metrics: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = arm_metrics.copy()
        df["model_name"] = df["model_name"].astype(str).str.strip()
        if df.empty:
            raise ValueError("Control vs treated overview requires objective arm metrics.")

        grouped = (
            df.assign(condition_group=np.where(df["is_control"], "Control", "Treated"))
            .groupby(["model_name", "condition_group"], as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                sycophancy_rate=("sycophancy_rate", "mean"),
            )
        )
        if grouped.empty:
            raise ValueError("No grouped control/treated metrics available.")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        palette = {"Control": "#4C78A8", "Treated": "#F58518"}
        metric_specs = [
            ("accuracy", "Control vs Treated Accuracy"),
            ("sycophancy_rate", "Control vs Treated Sycophancy"),
        ]

        for ax, (metric, title) in zip(axes, metric_specs):
            sns.barplot(
                data=grouped,
                x="model_name",
                y=metric,
                hue="condition_group",
                palette=palette,
                ax=ax,
            )
            ax.set_title(title, fontsize=13, pad=10)
            ax.set_xlabel("Model", fontsize=11)
            ax.set_ylabel("Rate", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True, title="")

        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_authority_level_main_effect_chart(
        self,
        arm_metrics: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = arm_metrics[~arm_metrics["is_control"]].copy()
        if df.empty:
            raise ValueError("Authority main-effect chart requires treated conditions.")

        acc_df = (
            df.groupby(["model_name", "authority_level"], as_index=False)["accuracy"]
            .mean()
            .dropna(subset=["authority_level"])
        )
        syc_df = (
            df[df["explicit_wrong_option"] == 1]
            .groupby(["model_name", "authority_level"], as_index=False)["sycophancy_rate"]
            .mean()
            .dropna(subset=["authority_level"])
        )
        if acc_df.empty:
            raise ValueError("No authority-level accuracy metrics available.")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
        for ax, metric_df, metric, title, marker in (
            (axes[0], acc_df, "accuracy", "Authority Main Effect on Accuracy", "o"),
            (axes[1], syc_df, "sycophancy_rate", "Authority Main Effect on Sycophancy", "s"),
        ):
            if metric_df.empty:
                continue
            sns.lineplot(
                data=metric_df,
                x="authority_level",
                y=metric,
                hue="model_name",
                style="model_name",
                markers=True,
                dashes=False,
                marker=marker,
                linewidth=2.2,
                ax=ax,
            )
            ax.set_title(title, fontsize=13, pad=10)
            ax.set_xlabel("Authority Level", fontsize=11)
            ax.set_ylabel("Rate", fontsize=11)
            ax.set_xticks([0, 1, 2])
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True, title="")

        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_confidence_level_main_effect_chart(
        self,
        arm_metrics: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = arm_metrics[(~arm_metrics["is_control"]) & (arm_metrics["explicit_wrong_option"] == 1)].copy()
        if df.empty:
            raise ValueError("Confidence main-effect chart requires treated wrong-option conditions.")

        acc_df = (
            df.groupby(["model_name", "confidence_level"], as_index=False)["accuracy"]
            .mean()
            .dropna(subset=["confidence_level"])
        )
        syc_df = (
            df.groupby(["model_name", "confidence_level"], as_index=False)["sycophancy_rate"]
            .mean()
            .dropna(subset=["confidence_level"])
        )
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
        for ax, metric_df, metric, title, marker in (
            (axes[0], acc_df, "accuracy", "Confidence Main Effect on Accuracy (Wrong Option Conditions)", "o"),
            (axes[1], syc_df, "sycophancy_rate", "Confidence Main Effect on Wrong-Option Follow (Wrong Option Conditions)", "s"),
        ):
            sns.lineplot(
                data=metric_df,
                x="confidence_level",
                y=metric,
                hue="model_name",
                style="model_name",
                markers=True,
                dashes=False,
                marker=marker,
                linewidth=2.2,
                ax=ax,
            )
            ax.set_title(title, fontsize=13, pad=10)
            ax.set_xlabel("Confidence Level", fontsize=11)
            ax.set_ylabel("Rate", fontsize=11)
            ax.set_xticks([0, 1])
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True, title="")

        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_wrong_option_exposure_main_effect_chart(
        self,
        arm_metrics: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = arm_metrics.copy()
        if df.empty:
            raise ValueError("Wrong-option exposure chart requires objective arm metrics.")

        grouped = (
            df.assign(condition_group=np.where(df["is_control"], "Control", "Treated"))
            .groupby(["model_name", "condition_group", "explicit_wrong_option"], as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                sycophancy_rate=("sycophancy_rate", "mean"),
            )
        )
        if grouped.empty:
            raise ValueError("No grouped wrong-option exposure metrics available.")

        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
        panel_specs = [
            ("Control", "accuracy", "Control: Accuracy by Wrong-Option Exposure"),
            ("Control", "sycophancy_rate", "Control: Wrong-Option Follow by Exposure"),
            ("Treated", "accuracy", "Treated: Accuracy by Wrong-Option Exposure"),
            ("Treated", "sycophancy_rate", "Treated: Wrong-Option Follow by Exposure"),
        ]
        for ax, (condition_group, metric, title) in zip(axes.flat, panel_specs):
            plot_df = grouped[grouped["condition_group"] == condition_group].copy()
            sns.barplot(
                data=plot_df,
                x="explicit_wrong_option",
                y=metric,
                hue="model_name",
                ax=ax,
            )
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel("Explicit Wrong Option", fontsize=11)
            ax.set_ylabel("Rate", fontsize=11)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0", "1"])
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True, title="")

        fig.tight_layout()
        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_condition_detail_chart(
        self,
        arm_metrics: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = arm_metrics.copy().sort_values(
            ["is_control", "authority_level", "confidence_level", "explicit_wrong_option", "condition_id"],
            na_position="first",
        )
        if df.empty:
            raise ValueError("Condition detail chart requires objective arm metrics.")

        df["display_label"] = df["condition_id"].astype(str)
        fig, axes = plt.subplots(1, 2, figsize=(max(14, len(df) * 0.7), 7), sharey=True)

        for ax, metric, title in (
            (axes[0], "accuracy", "Condition-Level Accuracy"),
            (axes[1], "sycophancy_rate", "Condition-Level Sycophancy"),
        ):
            sns.barplot(
                data=df,
                x="display_label",
                y=metric,
                hue="model_name",
                ax=ax,
            )
            ax.set_title(title, fontsize=13, pad=10)
            ax.set_xlabel("Condition ID", fontsize=11)
            ax.set_ylabel("Rate", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True, title="")
            for label in ax.get_xticklabels():
                label.set_ha("right")

        fig.tight_layout()
        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_model_baseline_vs_treated_overview(
        self,
        cross_model_summary: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = cross_model_summary.copy()
        melted = df.melt(
            id_vars=["model_name"],
            value_vars=["baseline_accuracy", "treated_accuracy_mean"],
            var_name="metric",
            value_name="accuracy",
        )
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.8), 6))
        sns.barplot(data=melted, x="model_name", y="accuracy", hue="metric", ax=ax)
        ax.set_title("Model Baseline vs Treated Accuracy", fontsize=14, pad=12)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True, title="")
        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_model_wrong_option_follow_comparison(
        self,
        cross_model_summary: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = cross_model_summary.copy()
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.8), 6))
        sns.barplot(data=df, x="model_name", y="treated_wrong_option_follow_rate_mean", hue="reasoning_mode", ax=ax)
        ax.set_title("Model Wrong-Option Follow Comparison", fontsize=14, pad=12)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Wrong-Option Follow Rate", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True, title="")
        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_reasoning_mode_group_comparison(
        self,
        cross_model_summary: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = (
            cross_model_summary.groupby("reasoning_mode", as_index=False)
            .agg(
                baseline_accuracy=("baseline_accuracy", "mean"),
                treated_accuracy_mean=("treated_accuracy_mean", "mean"),
                treated_wrong_option_follow_rate_mean=("treated_wrong_option_follow_rate_mean", "mean"),
            )
        )
        melted = df.melt(
            id_vars=["reasoning_mode"],
            value_vars=["baseline_accuracy", "treated_accuracy_mean", "treated_wrong_option_follow_rate_mean"],
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=melted, x="reasoning_mode", y="value", hue="metric", ax=ax)
        ax.set_title("Reasoning Mode Group Comparison", fontsize=14, pad=12)
        ax.set_xlabel("Reasoning Mode", fontsize=11)
        ax.set_ylabel("Rate", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=True, title="")
        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def plot_family_pair_condition_heatmap(
        self,
        cross_model_factor_metrics: pd.DataFrame,
        output_dir: Path,
        stem: str,
        formats: Sequence[str],
    ) -> Dict[str, Path]:
        df = cross_model_factor_metrics.copy()
        df["condition_id"] = (
            "a"
            + df["authority_level"].fillna(-1).astype(int).astype(str)
            + "_c"
            + df["confidence_level"].fillna(-1).astype(int).astype(str)
            + "_w"
            + df["explicit_wrong_option"].fillna(-1).astype(int).astype(str)
        )
        pivot = df.pivot_table(
            index="condition_id",
            columns="model_name",
            values="wrong_option_follow_rate",
            aggfunc="mean",
        ).fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.1), max(5, len(pivot.index) * 0.6)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0.0, vmax=1.0, ax=ax)
        ax.set_title("Family Pair Condition Heatmap (Wrong-Option Follow)", fontsize=14, pad=12)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Condition", fontsize=11)
        return self._save_figure(fig, output_dir, stem, formats, self.dpi)

    def run(
        self,
        input_file: Path,
        output_dir: Path,
        formats: Sequence[str] = ("png", "pdf"),
        valid_records_file: Optional[Path] = None,
        group_metrics_file: Optional[Path] = None,
        arm_metrics_file: Optional[Path] = None,
        category_arm_metrics_file: Optional[Path] = None,
        cross_model_summary_file: Optional[Path] = None,
        cross_model_factor_metrics_file: Optional[Path] = None,
    ) -> Dict[str, Path]:
        primary = self._load_csv(input_file)
        valid_records = self._as_valid_records(primary)
        group_metrics = self._as_group_metrics(primary)
        arm_metrics = self._as_arm_metrics(primary)
        category_arm_metrics = self._as_category_arm_metrics(primary)
        cross_model_summary = self._as_cross_model_summary(primary)
        cross_model_factor_metrics = self._as_cross_model_factor_metrics(primary)

        if valid_records is None and valid_records_file is not None:
            valid_records = self._as_valid_records(self._load_csv(valid_records_file))
        if group_metrics is None and group_metrics_file is not None:
            group_metrics = self._as_group_metrics(self._load_csv(group_metrics_file))
        if arm_metrics is None and arm_metrics_file is not None:
            arm_metrics = self._as_arm_metrics(self._load_csv(arm_metrics_file))
        if category_arm_metrics is None and category_arm_metrics_file is not None:
            category_arm_metrics = self._as_category_arm_metrics(
                self._load_csv(category_arm_metrics_file)
            )
        if cross_model_summary is None and cross_model_summary_file is not None:
            cross_model_summary = self._as_cross_model_summary(self._load_csv(cross_model_summary_file))
        if cross_model_factor_metrics is None and cross_model_factor_metrics_file is not None:
            cross_model_factor_metrics = self._as_cross_model_factor_metrics(
                self._load_csv(cross_model_factor_metrics_file)
            )

        if valid_records is None and arm_metrics is None:
            raise ValueError(
                "Could not find valid records columns or objective arm metrics. "
                "Use --valid-records-file or --arm-metrics-file."
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        delta_paths: Dict[str, Path] = {}
        boxplot_paths: Dict[str, Path] = {}
        control_overview_paths: Dict[str, Path] = {}
        authority_main_effect_paths: Dict[str, Path] = {}
        confidence_main_effect_paths: Dict[str, Path] = {}
        wrong_option_exposure_paths: Dict[str, Path] = {}
        condition_detail_paths: Dict[str, Path] = {}
        model_baseline_vs_treated_paths: Dict[str, Path] = {}
        model_wrong_option_follow_paths: Dict[str, Path] = {}
        reasoning_mode_group_paths: Dict[str, Path] = {}
        family_pair_heatmap_paths: Dict[str, Path] = {}
        rows_delta_models = 0

        if valid_records is not None:
            delta_table = self._build_delta_table(valid_records=valid_records, group_metrics=group_metrics)
            rows_delta_models = int(len(delta_table))
            delta_paths = self.plot_delta_bar_chart(
                delta_table=delta_table,
                output_dir=output_dir,
                stem=f"delta_bar_chart_{timestamp}",
                formats=formats,
            )
            boxplot_paths = self.plot_score_boxplot(
                valid_records=valid_records,
                output_dir=output_dir,
                stem=f"score_distribution_boxplot_{timestamp}",
                formats=formats,
            )
        if arm_metrics is not None:
            for plot_name, plot_fn, stem_name in (
                (
                    "control_vs_treated_overview",
                    self.plot_control_vs_treated_overview,
                    f"control_vs_treated_overview_{timestamp}",
                ),
                (
                    "authority_level_main_effect",
                    self.plot_authority_level_main_effect_chart,
                    f"authority_level_main_effect_{timestamp}",
                ),
                (
                    "confidence_level_main_effect",
                    self.plot_confidence_level_main_effect_chart,
                    f"confidence_level_main_effect_{timestamp}",
                ),
                (
                    "wrong_option_exposure_main_effect",
                    self.plot_wrong_option_exposure_main_effect_chart,
                    f"wrong_option_exposure_main_effect_{timestamp}",
                ),
                (
                    "condition_id_detail",
                    self.plot_condition_detail_chart,
                    f"condition_id_detail_{timestamp}",
                ),
            ):
                try:
                    saved_plot = plot_fn(
                        arm_metrics=arm_metrics,
                        output_dir=output_dir,
                        stem=stem_name,
                        formats=formats,
                    )
                except Exception as exc:
                    logger.warning("Skip %s plot: %s", plot_name, exc)
                    saved_plot = {}
                if plot_name == "control_vs_treated_overview":
                    control_overview_paths = saved_plot
                elif plot_name == "authority_level_main_effect":
                    authority_main_effect_paths = saved_plot
                elif plot_name == "confidence_level_main_effect":
                    confidence_main_effect_paths = saved_plot
                elif plot_name == "wrong_option_exposure_main_effect":
                    wrong_option_exposure_paths = saved_plot
                elif plot_name == "condition_id_detail":
                    condition_detail_paths = saved_plot
        if cross_model_summary is not None and not cross_model_summary.empty:
            for plot_name, plot_fn, stem_name in (
                (
                    "model_baseline_vs_treated_overview",
                    self.plot_model_baseline_vs_treated_overview,
                    f"model_baseline_vs_treated_overview_{timestamp}",
                ),
                (
                    "model_wrong_option_follow_comparison",
                    self.plot_model_wrong_option_follow_comparison,
                    f"model_wrong_option_follow_comparison_{timestamp}",
                ),
                (
                    "reasoning_mode_group_comparison",
                    self.plot_reasoning_mode_group_comparison,
                    f"reasoning_mode_group_comparison_{timestamp}",
                ),
            ):
                try:
                    saved_plot = plot_fn(
                        cross_model_summary=cross_model_summary,
                        output_dir=output_dir,
                        stem=stem_name,
                        formats=formats,
                    )
                except Exception as exc:
                    logger.warning("Skip %s plot: %s", plot_name, exc)
                    saved_plot = {}
                if plot_name == "model_baseline_vs_treated_overview":
                    model_baseline_vs_treated_paths = saved_plot
                elif plot_name == "model_wrong_option_follow_comparison":
                    model_wrong_option_follow_paths = saved_plot
                elif plot_name == "reasoning_mode_group_comparison":
                    reasoning_mode_group_paths = saved_plot
        if cross_model_factor_metrics is not None and not cross_model_factor_metrics.empty:
            try:
                family_pair_heatmap_paths = self.plot_family_pair_condition_heatmap(
                    cross_model_factor_metrics=cross_model_factor_metrics,
                    output_dir=output_dir,
                    stem=f"family_pair_condition_heatmap_{timestamp}",
                    formats=formats,
                )
            except Exception as exc:
                logger.warning("Skip family_pair_condition_heatmap plot: %s", exc)

        summary_path = output_dir / f"visualization_summary_{timestamp}.json"
        summary_payload = {
            "generated_at": datetime.now().isoformat(),
            "input_file": str(input_file),
            "valid_records_file": str(valid_records_file) if valid_records_file else "",
            "group_metrics_file": str(group_metrics_file) if group_metrics_file else "",
            "arm_metrics_file": str(arm_metrics_file) if arm_metrics_file else "",
            "category_arm_metrics_file": str(category_arm_metrics_file) if category_arm_metrics_file else "",
            "rows_valid_records": int(len(valid_records)) if valid_records is not None else 0,
            "rows_delta_models": rows_delta_models,
            "rows_arm_metrics": int(len(arm_metrics)) if arm_metrics is not None else 0,
            "rows_category_arm_metrics": int(len(category_arm_metrics)) if category_arm_metrics is not None else 0,
            "rows_cross_model_summary": int(len(cross_model_summary)) if cross_model_summary is not None else 0,
            "rows_cross_model_factor_metrics": int(len(cross_model_factor_metrics)) if cross_model_factor_metrics is not None else 0,
            "files": {
                "delta_bar_chart": {k: str(v) for k, v in delta_paths.items()},
                "score_distribution_boxplot": {k: str(v) for k, v in boxplot_paths.items()},
                "control_vs_treated_overview": {k: str(v) for k, v in control_overview_paths.items()},
                "authority_level_main_effect": {k: str(v) for k, v in authority_main_effect_paths.items()},
                "confidence_level_main_effect": {k: str(v) for k, v in confidence_main_effect_paths.items()},
                "wrong_option_exposure_main_effect": {k: str(v) for k, v in wrong_option_exposure_paths.items()},
                "condition_id_detail": {k: str(v) for k, v in condition_detail_paths.items()},
                "model_baseline_vs_treated_overview": {k: str(v) for k, v in model_baseline_vs_treated_paths.items()},
                "model_wrong_option_follow_comparison": {k: str(v) for k, v in model_wrong_option_follow_paths.items()},
                "reasoning_mode_group_comparison": {k: str(v) for k, v in reasoning_mode_group_paths.items()},
                "family_pair_condition_heatmap": {k: str(v) for k, v in family_pair_heatmap_paths.items()},
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        saved: Dict[str, Path] = {"summary_json": summary_path}
        for ext, path in delta_paths.items():
            saved[f"delta_bar_chart_{ext}"] = path
        for ext, path in boxplot_paths.items():
            saved[f"score_distribution_boxplot_{ext}"] = path
        for ext, path in control_overview_paths.items():
            saved[f"control_vs_treated_overview_{ext}"] = path
        for ext, path in authority_main_effect_paths.items():
            saved[f"authority_level_main_effect_{ext}"] = path
        for ext, path in confidence_main_effect_paths.items():
            saved[f"confidence_level_main_effect_{ext}"] = path
        for ext, path in wrong_option_exposure_paths.items():
            saved[f"wrong_option_exposure_main_effect_{ext}"] = path
        for ext, path in condition_detail_paths.items():
            saved[f"condition_id_detail_{ext}"] = path
        for ext, path in model_baseline_vs_treated_paths.items():
            saved[f"model_baseline_vs_treated_overview_{ext}"] = path
        for ext, path in model_wrong_option_follow_paths.items():
            saved[f"model_wrong_option_follow_comparison_{ext}"] = path
        for ext, path in reasoning_mode_group_paths.items():
            saved[f"reasoning_mode_group_comparison_{ext}"] = path
        for ext, path in family_pair_heatmap_paths.items():
            saved[f"family_pair_condition_heatmap_{ext}"] = path

        logger.info("Visualization finished. Outputs saved to %s", output_dir)
        return saved
