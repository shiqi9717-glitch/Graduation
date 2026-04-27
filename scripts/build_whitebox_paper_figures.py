#!/usr/bin/env python3
"""Build paper-ready figures from frozen white-box figure asset CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "outputs/experiments/whitebox_mechanistic_figure_assets/20260426_230559"
OUT_DIR = ROOT / "docs/papers/figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def build_figure2() -> None:
    rows = _read_csv(ASSET_DIR / "figure2_qwen_mainline_effect_ci.csv")
    metrics = [
        ("stance_drift_delta", "Stance drift\n(delta)"),
        ("pressured_compliance_delta", "Pressured compliance\n(delta)"),
        ("recovery_delta", "Recovery\n(delta)"),
        ("baseline_damage_rate", "Baseline damage\n(rate)"),
    ]
    models = ["Qwen-7B main baseline", "Qwen-3B replication"]
    colors = {"Qwen-7B main baseline": "#2E6FBB", "Qwen-3B replication": "#D9822B"}

    lookup = {(r["model_group"], r["metric_key"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.34
    x = list(range(len(metrics)))

    for offset, model in [(-width / 2, models[0]), (width / 2, models[1])]:
        vals, lows, highs = [], [], []
        for key, _ in metrics:
            row = lookup[(model, key)]
            est = float(row["estimate"])
            vals.append(est)
            lows.append(est - float(row["ci_low"]))
            highs.append(float(row["ci_high"]) - est)
        xpos = [i + offset for i in x]
        ax.bar(xpos, vals, width=width, color=colors[model], label=model.replace(" main baseline", ""))
        ax.errorbar(xpos, vals, yerr=[lows, highs], fmt="none", ecolor="#222222", elinewidth=1.1, capsize=3)

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Effect estimate")
    ax.set_title("Qwen mainline intervention effect with 95% bootstrap CIs")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    _save(fig, "figure2_qwen_mainline_effect_ci.pdf")


def _tradeoff_panel(ax: plt.Axes, rows: list[dict[str, str]], title: str) -> None:
    for row in rows:
        x = float(row["baseline_damage_rate"])
        y = -float(row["pressured_compliance_delta"])
        xerr = [[x - float(row["baseline_damage_ci_low"])], [float(row["baseline_damage_ci_high"]) - x]]
        yerr = [[y - (-float(row["pressured_compliance_ci_high"]))], [(-float(row["pressured_compliance_ci_low"])) - y]]
        label = row["model_group"].replace(" secondary causal confirmation", "").replace(" cross-family positive replication", "")
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=3, markersize=6)
        ax.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="#999999", linewidth=0.7)
    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.set_xlabel("Baseline damage rate")
    ax.set_ylabel("Compliance reduction\n(-delta)")
    ax.set_title(title)
    ax.grid(color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def build_figure3() -> None:
    panel_a = _read_csv(ASSET_DIR / "figure3_panelA_qwen_proxy_tradeoff.csv")
    panel_b = _read_csv(ASSET_DIR / "figure3_panelB_bridge_causal_tradeoff.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    _tradeoff_panel(axes[0], panel_a, "A. Qwen objective-local proxy")
    _tradeoff_panel(axes[1], panel_b, "B. Bridge causal / transfer-style")
    fig.suptitle("Cross-model tradeoff summary (split by metric family)", y=1.02)
    _save(fig, "figure3_split_panel_tradeoff.pdf")


def build_figure4() -> None:
    rows = _read_csv(ASSET_DIR / "figure4_llama_projection_summary.csv")
    values = {row["metric"]: float(row["value"]) for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].bar(
        ["Baseline\nprojection", "Pressured\nprojection"],
        [values["mean_baseline_projection_norm"], values["mean_pressured_projection_norm"]],
        color=["#6C7A89", "#2E6FBB"],
    )
    axes[0].set_ylabel("Projection norm")
    axes[0].set_title("Representation-level localization")
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axes[0].set_axisbelow(True)

    behavior_labels = ["Drift\nreduction", "Compliance\nreduction", "Recovery\nchange", "Baseline\ndamage"]
    behavior_values = [0.08240833333333333, -0.0, -0.084725, 0.24994166666666667]
    axes[1].bar(behavior_labels, behavior_values, color=["#2E6FBB", "#9AA5B1", "#C95C54", "#C95C54"])
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Behavioral effect")
    axes[1].set_title("Weak behavioral controllability")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axes[1].set_axisbelow(True)

    fig.suptitle("Llama: locatable pressure direction, weak intervention transfer", y=1.03)
    _save(fig, "figure4_llama_localization_vs_controllability.pdf")


def main() -> None:
    build_figure2()
    build_figure3()
    build_figure4()


if __name__ == "__main__":
    main()
