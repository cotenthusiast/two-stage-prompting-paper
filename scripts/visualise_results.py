"""Standalone visualisation script for two-stage-prompting eval outputs.

Reads CSVs from reports/<run_id>/<benchmark>/ and writes plots to
reports/<run_id>/plots/<benchmark>/.  Does not touch any eval logic.

Usage:
    python scripts/visualise_results.py --run-id <run_id> --benchmark mmlu
    python scripts/visualise_results.py --run-id <run_id> --benchmark arc

    # Paper eval (reads from paper_results/reports/):
    python scripts/visualise_results.py --run-id paper_api_main --benchmark mmlu \\
        --config paper_results/eval_config.yaml
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REPORTS_DIR = _ROOT / "reports"

_BENCHMARK_ALIASES = {"arc": "arc_challenge"}

METHOD_ORDER = [
    "baseline",
    "two_prompt",
    "two_prompt_v2",
    "two_prompt_v3",
    "cyclic",
    "pride",
    "text_extraction",
    "twostage_semantic_match",
    "twostage_semantic_match_v2",
]

MODEL_ORDER = [
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "llama-3.1-8b-instant",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

_MODEL_LABELS: dict[str, str] = {
    "gpt-4.1-mini": "GPT-4.1\nmini",
    "gemini-2.5-flash": "Gemini\n2.5 Flash",
    "llama-3.1-8b-instant": "Llama-3.1\n8B (Groq)",
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "Qwen2.5\n7B-Turbo",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5\n7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1\n8B",
}

_METHOD_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "two_prompt": "Two-Stage",
    "two_prompt_v2": "Two-Stage\nv2",
    "two_prompt_v3": "Two-Stage\nv3",
    "cyclic": "Cyclic",
    "two_prompt_cyclic": "Two-Stage\nCyclic",
    "pride": "PriDe",
    "text_extraction": "Text\nExtract",
    "twostage_semantic_match": "Semantic\nMatch",
    "twostage_semantic_match_v2": "Semantic\nMatch v2",
}

_METHOD_COLORS: dict[str, str] = {
    "baseline": "#4878cf",
    "two_prompt": "#e87b22",
    "two_prompt_v2": "#f0a050",
    "two_prompt_v3": "#f5c580",
    "cyclic": "#3dab5e",
    "two_prompt_cyclic": "#9067b8",
    "pride": "#c33c3c",
    "text_extraction": "#d4a017",
    "twostage_semantic_match": "#2196a8",
    "twostage_semantic_match_v2": "#5bbac8",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[warn] not found, skipping plot: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[warn] empty CSV, skipping plot: {path}")
        return None
    return df


def _present(df: pd.DataFrame, col: str, order: list[str]) -> list[str]:
    present = set(df[col].dropna().astype(str).unique())
    return [x for x in order if x in present]


def _mlabel(m: str) -> str:
    return _MODEL_LABELS.get(m, m)


def _methlabel(m: str) -> str:
    return _METHOD_LABELS.get(m, m)


def _sig(p: float) -> str:
    if not isinstance(p, float) or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _clean_bar_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_axisbelow(False)


def _row_for_model(df: pd.DataFrame, model: str) -> pd.Series | None:
    rows = df[df["model"].astype(str) == model]
    if rows.empty:
        return None
    return rows.iloc[0]


def _bar_positions(n_methods: int) -> tuple[np.ndarray, float]:
    width = min(0.8 / n_methods, 0.25)
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width
    return offsets, width


# ── Plot 1: End-to-end accuracy bar chart ─────────────────────────────────────

def plot_accuracy(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    df = _load_csv(report_dir / "accuracy.csv")
    if df is None:
        return

    models = _present(df, "model", MODEL_ORDER)
    methods = _present(df, "method", METHOD_ORDER)
    if not models or not methods:
        return

    n_m = len(models)
    n_k = len(methods)
    x = np.arange(n_m)
    offsets, width = _bar_positions(n_k)

    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.6 + 1), 5))

    for i, method in enumerate(methods):
        mdf = df[df["method"].astype(str) == method]
        vals, err_lo, err_hi = [], [], []
        for model in models:
            row = _row_for_model(mdf, model)
            if row is not None:
                v = float(row["end_to_end_accuracy"])
                lo = max(0.0, v - float(row["end_to_end_accuracy_ci_lower"]))
                hi = max(0.0, float(row["end_to_end_accuracy_ci_upper"]) - v)
            else:
                v, lo, hi = np.nan, np.nan, np.nan
            vals.append(v)
            err_lo.append(lo)
            err_hi.append(hi)

        ax.bar(
            x + offsets[i], vals, width,
            color=_METHOD_COLORS.get(method, f"C{i}"),
            label=_methlabel(method),
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6},
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_mlabel(m) for m in models], ha="center")
    ax.set_ylabel("End-to-end accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"End-to-end accuracy — {bm_label}")
    ax.legend(frameon=False, loc="upper right", ncol=min(n_k, 3))
    _clean_bar_ax(ax)

    fig.tight_layout()
    out = plots_dir / "accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 2: Conditional accuracy bar chart ────────────────────────────────────

def plot_conditional_accuracy(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    df = _load_csv(report_dir / "accuracy.csv")
    if df is None:
        return
    if "conditional_accuracy" not in df.columns:
        return

    models = _present(df, "model", MODEL_ORDER)
    methods = _present(df, "method", METHOD_ORDER)
    if not models or not methods:
        return

    n_m = len(models)
    n_k = len(methods)
    x = np.arange(n_m)
    offsets, width = _bar_positions(n_k)

    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.6 + 1), 5))

    for i, method in enumerate(methods):
        mdf = df[df["method"].astype(str) == method]
        vals, err_lo, err_hi = [], [], []
        for model in models:
            row = _row_for_model(mdf, model)
            if row is not None and not pd.isna(row.get("conditional_accuracy")):
                v = float(row["conditional_accuracy"])
                lo = max(0.0, v - float(row["conditional_accuracy_ci_lower"]))
                hi = max(0.0, float(row["conditional_accuracy_ci_upper"]) - v)
            else:
                v, lo, hi = np.nan, np.nan, np.nan
            vals.append(v)
            err_lo.append(lo)
            err_hi.append(hi)

        ax.bar(
            x + offsets[i], vals, width,
            color=_METHOD_COLORS.get(method, f"C{i}"),
            label=_methlabel(method),
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6},
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_mlabel(m) for m in models], ha="center")
    ax.set_ylabel("Conditional accuracy (scored rows only)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Conditional accuracy (excludes unscorable) — {bm_label}")
    ax.legend(frameon=False, loc="upper right", ncol=min(n_k, 3))
    _clean_bar_ax(ax)

    fig.tight_layout()
    out = plots_dir / "conditional_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 4: Bias-from-uniform bar chart ───────────────────────────────────────

def plot_bias_from_uniform(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    df = _load_csv(report_dir / "positional_bias.csv")
    if df is None:
        return

    models = _present(df, "model", MODEL_ORDER)
    methods = _present(df, "method", METHOD_ORDER)
    if not models or not methods:
        return

    n_m = len(models)
    n_k = len(methods)
    x = np.arange(n_m)
    offsets, width = _bar_positions(n_k)

    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.6 + 1), 5))

    for i, method in enumerate(methods):
        mdf = df[df["method"].astype(str) == method]
        vals, err_lo, err_hi = [], [], []
        for model in models:
            row = _row_for_model(mdf, model)
            if row is not None:
                v = float(row["bias_from_uniform"])
                lo = max(0.0, v - float(row["bias_from_uniform_ci_lower"]))
                hi = max(0.0, float(row["bias_from_uniform_ci_upper"]) - v)
            else:
                v, lo, hi = np.nan, np.nan, np.nan
            vals.append(v)
            err_lo.append(lo)
            err_hi.append(hi)

        ax.bar(
            x + offsets[i], vals, width,
            color=_METHOD_COLORS.get(method, f"C{i}"),
            label=_methlabel(method),
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6},
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_mlabel(m) for m in models], ha="center")
    ax.set_ylabel("Bias from uniform")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Positional bias (deviation from uniform) — {bm_label}")
    ax.legend(frameon=False, loc="upper right", ncol=min(n_k, 3))
    _clean_bar_ax(ax)

    fig.tight_layout()
    out = plots_dir / "bias_from_uniform.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 5: MAD bar chart ──────────────────────────────────────────────────────

def plot_mad(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    df = _load_csv(report_dir / "positional_bias.csv")
    if df is None:
        return

    models = _present(df, "model", MODEL_ORDER)
    methods = _present(df, "method", METHOD_ORDER)
    if not models or not methods:
        return

    n_m = len(models)
    n_k = len(methods)
    x = np.arange(n_m)
    offsets, width = _bar_positions(n_k)

    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.6 + 1), 5))

    for i, method in enumerate(methods):
        mdf = df[df["method"].astype(str) == method]
        vals, err_lo, err_hi = [], [], []
        for model in models:
            row = _row_for_model(mdf, model)
            if row is not None:
                v = float(row["mean_abs_deviation"])
                lo = max(0.0, v - float(row["mean_abs_deviation_ci_lower"]))
                hi = max(0.0, float(row["mean_abs_deviation_ci_upper"]) - v)
            else:
                v, lo, hi = np.nan, np.nan, np.nan
            vals.append(v)
            err_lo.append(lo)
            err_hi.append(hi)

        ax.bar(
            x + offsets[i], vals, width,
            color=_METHOD_COLORS.get(method, f"C{i}"),
            label=_methlabel(method),
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 0.8, "ecolor": "black", "alpha": 0.6},
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_mlabel(m) for m in models], ha="center")
    ax.set_ylabel("Mean absolute deviation (%pts)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"MAD from ground-truth answer distribution — {bm_label}")
    ax.legend(frameon=False, loc="upper right", ncol=min(n_k, 3))
    _clean_bar_ax(ax)

    fig.tight_layout()
    out = plots_dir / "mad.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 6: Broken / fixed chart ──────────────────────────────────────────────

def plot_broken_fixed(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    df = _load_csv(report_dir / "overlap.csv")
    if df is None:
        return

    models = _present(df, "model", MODEL_ORDER)
    # overlap.csv only has non-baseline methods
    methods = _present(df, "method", [m for m in METHOD_ORDER if m != "baseline"])
    if not models or not methods:
        return

    n_cols = min(2, len(models))
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 4),
        squeeze=False,
    )

    bar_w = 0.35
    x = np.arange(len(methods))
    broken_color = "#d94f4f"
    fixed_color = "#4caf82"

    for idx, model in enumerate(models):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri][ci]

        mdf = df[df["model"].astype(str) == model]
        broken_vals, fixed_vals, net_vals, sig_vals = [], [], [], []

        for method in methods:
            row = _row_for_model(mdf, method) if "method" in mdf.columns else None
            rows_m = mdf[mdf["method"].astype(str) == method]
            if not rows_m.empty:
                r = rows_m.iloc[0]
                broken_vals.append(int(r["baseline_only_correct"]))
                fixed_vals.append(int(r["method_only_correct"]))
                net_vals.append(int(r["net_effect"]))
                sig_vals.append(_sig(float(r["mcnemar_p"])))
            else:
                broken_vals.append(0)
                fixed_vals.append(0)
                net_vals.append(0)
                sig_vals.append("")

        ax.bar(x - bar_w / 2, broken_vals, bar_w, color=broken_color, label="Broken", zorder=3)
        ax.bar(x + bar_w / 2, fixed_vals, bar_w, color=fixed_color, label="Fixed", zorder=3)

        # Annotation: net effect + significance above each bar pair
        ymax = max(max(broken_vals, default=0), max(fixed_vals, default=0), 1)
        ax.set_ylim(0, ymax * 1.3)
        for j, (net, sig) in enumerate(zip(net_vals, sig_vals)):
            sign = "+" if net >= 0 else ""
            ax.text(
                x[j], ymax * 1.08,
                f"{sign}{net}{sig}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color="#2d7d46" if net > 0 else ("#c0392b" if net < 0 else "#555"),
            )

        ax.set_xticks(x)
        ax.set_xticklabels([_methlabel(m) for m in methods], ha="center", fontsize=8)
        ax.set_title(_mlabel(model).replace("\n", " "), fontsize=10)
        ax.set_ylabel("Questions", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        _clean_bar_ax(ax)

    for idx in range(len(models), n_rows * n_cols):
        ri, ci = divmod(idx, n_cols)
        axes[ri][ci].set_visible(False)

    fig.suptitle(
        f"Questions broken vs. fixed relative to baseline — {bm_label}\n"
        "Net = fixed − broken   |   * p<0.05  ** p<0.01  *** p<0.001 (McNemar)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = plots_dir / "broken_fixed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 7: Free-text decomposition heatmap ───────────────────────────────────

def _plot_ft_decomp_for_method(
    method_df: pd.DataFrame,
    method_label: str,
    plots_dir: Path,
    bm_label: str,
    filename: str,
) -> None:
    """Render and save one free-text decomposition figure for a single method."""
    col_labels = ["Match\ncorrect", "Match\nwrong"]
    row_labels = ["FT\ncorrect", "FT\nwrong"]

    method_models = _present(method_df, "model", MODEL_ORDER)
    if not method_models:
        return

    n_cols = min(2, len(method_models))
    n_rows = math.ceil(len(method_models) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3.5),
        squeeze=False,
    )

    for idx, model in enumerate(method_models):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri][ci]

        mdf = method_df[method_df["model"].astype(str) == model]
        if mdf.empty:
            ax.set_visible(False)
            continue

        r = mdf.iloc[0]
        total = max(int(r["n_total"]), 1)

        grid = np.array([
            [r["ft_correct_match_correct"], r["ft_correct_match_wrong"]],
            [r["ft_wrong_match_correct"],   r["ft_wrong_match_wrong"]],
        ], dtype=float)

        ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=grid.max() or 1)

        gmax = grid.max() or 1
        for row_i in range(2):
            for col_i in range(2):
                count = int(grid[row_i, col_i])
                pct = grid[row_i, col_i] / total * 100
                text_color = "white" if grid[row_i, col_i] > gmax * 0.55 else "black"
                ax.text(
                    col_i, row_i,
                    f"{count}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color=text_color,
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(_mlabel(model).replace("\n", " "), fontsize=10)

    for idx in range(len(method_models), n_rows * n_cols):
        ri, ci = divmod(idx, n_cols)
        axes[ri][ci].set_visible(False)

    fig.suptitle(
        f"Free-text stage decomposition — {method_label} — {bm_label}",
        fontsize=11,
    )
    fig.tight_layout()
    out = plots_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_ft_decomposition(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    """One figure per method — handles both old (no method column) and new CSV schemas."""
    df = _load_csv(report_dir / "free_text_decomposition.csv")
    if df is None:
        return

    if "method" in df.columns:
        # New schema: one row per (method, model) — produce one file per method.
        for method in _present(df, "method", METHOD_ORDER):
            method_df = df[df["method"].astype(str) == method]
            _plot_ft_decomp_for_method(
                method_df,
                method_label=_methlabel(method).replace("\n", " "),
                plots_dir=plots_dir,
                bm_label=bm_label,
                filename=f"ft_decomposition_{method}.png",
            )
    else:
        # Old schema: one row per model, no method column.
        _plot_ft_decomp_for_method(
            df,
            method_label="Two-Stage",
            plots_dir=plots_dir,
            bm_label=bm_label,
            filename="ft_decomposition.png",
        )


# ── Plot 8: Unscorable-rate heatmap ───────────────────────────────────────────

def plot_parse_failure_heatmap(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    """Method × model heatmap of final_unscorable_rate.

    Useful for diagnosing text_extraction and twostage_semantic_match, which
    can have higher parse failure rates than the letter-extraction methods.
    """
    df = _load_csv(report_dir / "accuracy.csv")
    if df is None:
        return
    if "final_unscorable_rate" not in df.columns:
        return

    methods = _present(df, "method", METHOD_ORDER)
    models = _present(df, "model", MODEL_ORDER)
    if not methods or not models:
        return

    grid = np.full((len(methods), len(models)), np.nan)
    for ri, method in enumerate(methods):
        mdf = df[df["method"].astype(str) == method]
        for ci, model in enumerate(models):
            row = _row_for_model(mdf, model)
            if row is not None and not pd.isna(row.get("final_unscorable_rate")):
                grid[ri, ci] = float(row["final_unscorable_rate"]) * 100

    cell_w = max(1.5, len(models) * 1.3)
    cell_h = max(1.5, len(methods) * 0.7)
    fig, ax = plt.subplots(figsize=(cell_w + 1.5, cell_h + 1))

    vmax = max(float(np.nanmax(grid)), 1.0) if not np.all(np.isnan(grid)) else 1.0
    im = ax.imshow(grid, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)

    for ri in range(len(methods)):
        for ci in range(len(models)):
            val = grid[ri, ci]
            text = f"{val:.1f}%" if not np.isnan(val) else "—"
            text_color = "white" if (not np.isnan(val) and val > vmax * 0.6) else "black"
            ax.text(ci, ri, text, ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(
        [_mlabel(m).replace("\n", " ") for m in models],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([_methlabel(m).replace("\n", " ") for m in methods], fontsize=8)
    ax.set_title(f"Unscorable rate (%) — {bm_label}", fontsize=11)
    plt.colorbar(im, ax=ax, label="Unscorable %", fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = plots_dir / "parse_failure_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Plot 9: E2E vs conditional accuracy gap ────────────────────────────────────

def plot_accuracy_gap(report_dir: Path, plots_dir: Path, bm_label: str) -> None:
    """Per-model panel: e2e accuracy and conditional accuracy overlaid.

    The gap between the two bars shows how much parse failures cost each method.
    Methods with high unscorable rates will show a large gap.
    """
    df = _load_csv(report_dir / "accuracy.csv")
    if df is None:
        return
    if "conditional_accuracy" not in df.columns:
        return

    models = _present(df, "model", MODEL_ORDER)
    methods = _present(df, "method", METHOD_ORDER)
    if not models or not methods:
        return

    n_cols = min(2, len(models))
    n_rows = math.ceil(len(models) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 3.8),
        squeeze=False,
    )

    x = np.arange(len(methods))
    bar_w = 0.35

    for idx, model in enumerate(models):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri][ci]

        mdf = df[df["model"].astype(str) == model]
        e2e_vals, cond_vals = [], []

        for method in methods:
            rows_m = mdf[mdf["method"].astype(str) == method]
            if not rows_m.empty:
                r = rows_m.iloc[0]
                e2e_vals.append(float(r["end_to_end_accuracy"]) if not pd.isna(r.get("end_to_end_accuracy")) else np.nan)
                cond_vals.append(float(r["conditional_accuracy"]) if not pd.isna(r.get("conditional_accuracy")) else np.nan)
            else:
                e2e_vals.append(np.nan)
                cond_vals.append(np.nan)

        ax.bar(x - bar_w / 2, e2e_vals, bar_w, label="End-to-end", color="#4878cf", alpha=0.85, zorder=3)
        ax.bar(x + bar_w / 2, cond_vals, bar_w, label="Conditional", color="#e87b22", alpha=0.85, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([_methlabel(m).replace("\n", " ") for m in methods], rotation=25, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_title(_mlabel(model).replace("\n", " "), fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _clean_bar_ax(ax)

    for idx in range(len(models), n_rows * n_cols):
        ri, ci = divmod(idx, n_cols)
        axes[ri][ci].set_visible(False)

    fig.suptitle(
        f"End-to-end vs conditional accuracy — {bm_label}\n"
        "(gap = cost of unscorable outputs)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out = plots_dir / "accuracy_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise eval CSVs for one experiment run.",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Run ID — folder name under the reports directory.",
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Benchmark subfolder (e.g. mmlu, arc, arc_challenge).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config to override reports_dir (e.g. paper_results/eval_config.yaml).",
    )
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text())
        reports_dir = _ROOT / cfg["paths"]["reports_dir"]
    else:
        reports_dir = _REPORTS_DIR

    benchmark = _BENCHMARK_ALIASES.get(args.benchmark or "", args.benchmark)

    report_dir = reports_dir / args.run_id
    if benchmark:
        report_dir = report_dir / benchmark

    if not report_dir.exists():
        print(f"[error] report dir not found: {report_dir}")
        return

    plots_dir = reports_dir / args.run_id / "plots"
    if benchmark:
        plots_dir = plots_dir / benchmark
    plots_dir.mkdir(parents=True, exist_ok=True)

    bm_label = benchmark or args.run_id
    print(f"[viz] Reading from {report_dir}")
    print(f"[viz] Writing plots to {plots_dir}")

    plot_accuracy(report_dir, plots_dir, bm_label)
    plot_conditional_accuracy(report_dir, plots_dir, bm_label)
    plot_accuracy_gap(report_dir, plots_dir, bm_label)
    plot_parse_failure_heatmap(report_dir, plots_dir, bm_label)
    plot_bias_from_uniform(report_dir, plots_dir, bm_label)
    plot_mad(report_dir, plots_dir, bm_label)
    plot_broken_fixed(report_dir, plots_dir, bm_label)
    plot_ft_decomposition(report_dir, plots_dir, bm_label)

    print("[viz] Done.")


if __name__ == "__main__":
    main()
