"""Baseline fallback analysis — separate from the main evaluation pipeline.

For each unscorable row produced by a target method, asks: what accuracy would
we observe if we substituted the corresponding baseline prediction (same model,
same question, same benchmark)?

This is a diagnostic table, not a main result. It does NOT modify any input
CSVs and writes only to paper_results/reports_fallback/.

Usage:
    # Folder that already contains baseline rows (main eval folders):
    python scripts/fallback_analysis.py \\
        --run-dir paper_results/eval_ready/paper_api_main \\
        --benchmark mmlu

    # Ablation folder with no baseline rows — point --baseline-dir at a main folder:
    python scripts/fallback_analysis.py \\
        --run-dir paper_results/eval_ready/paper_api_ablation_v2 \\
        --baseline-dir paper_results/eval_ready/paper_api_main \\
        --benchmark mmlu

Output:
    paper_results/reports_fallback/<folder_name>/<benchmark>/
        fallback_summary.csv      — per (model, method) summary row
        fallback_per_question.csv — per (question, model, method) detail row
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_ROOT = _ROOT / "paper_results" / "reports_fallback"

_BENCHMARK_ALIASES = {"arc": "arc_challenge"}

# Methods for which fallback is meaningful: they have a free-text / extraction
# stage that can fail to produce a parseable letter answer.
# cyclic and pride are excluded: cyclic uses majority vote so "unscorable" means
# all four permutations failed; pride has its own logprob-based scoring.
FALLBACK_TARGET_METHODS = [
    "two_prompt",
    "two_prompt_v2",
    "two_prompt_v3",
    "text_extraction",
    "twostage_semantic_match",
    "twostage_semantic_match_v2",
]


def _load_csvs(folder: Path, benchmark: str | None) -> pd.DataFrame:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {folder}")
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    if benchmark and "benchmark" in df.columns:
        df = df[df["benchmark"] == benchmark].reset_index(drop=True)
    return df


def _baseline_index(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Return baseline rows indexed by (model_name, question_id)."""
    bl = baseline_df[baseline_df["method_name"] == "baseline"].copy()
    bl = bl.drop_duplicates(subset=["model_name", "question_id"])
    return bl.set_index(["model_name", "question_id"])


def run_fallback_analysis(
    run_df: pd.DataFrame,
    bl_index: pd.DataFrame,
    benchmark: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fallback metrics.

    Returns (summary_df, per_question_df). Both are empty DataFrames if there
    is nothing to compute.
    """
    bm_label = benchmark or "unknown"
    summary_rows: list[dict] = []
    pq_rows: list[dict] = []

    models = sorted(run_df["model_name"].dropna().unique())

    for method in FALLBACK_TARGET_METHODS:
        method_df = run_df[run_df["method_name"] == method]
        if method_df.empty:
            continue

        for model in models:
            group = method_df[method_df["model_name"] == model]
            if group.empty:
                continue

            total = len(group)
            original_scored = int(group["is_correct"].notna().sum())
            original_correct = int(group["is_correct"].eq(True).sum())
            original_e2e = original_correct / total if total > 0 else 0.0
            original_cond = (
                original_correct / original_scored if original_scored > 0 else float("nan")
            )

            # Eligible for fallback: API succeeded but is_correct is NaN
            unscorable_mask = (
                (group["model_status"].fillna("") != "failure")
                & group["is_correct"].isna()
            )
            unscorable_count = int(unscorable_mask.sum())
            unscorable_rate = unscorable_count / total if total > 0 else 0.0

            fallback_available = 0
            fallback_correct = 0
            no_baseline_count = 0

            for _, row in group[unscorable_mask].iterrows():
                bl_key = (model, row["question_id"])
                if bl_key in bl_index.index:
                    bl_row = bl_index.loc[bl_key]
                    fallback_available += 1
                    bl_correct = bl_row["is_correct"]
                    pq_rows.append({
                        "question_id": row["question_id"],
                        "model_name": model,
                        "benchmark": bm_label,
                        "method_name": method,
                        "original_parsed_choice": row.get("parsed_choice"),
                        "original_is_correct": row["is_correct"],
                        "baseline_parsed_choice": bl_row.get("parsed_choice"),
                        "baseline_is_correct": bl_correct,
                        "fallback_used": True,
                        "fallback_is_correct": bl_correct,
                    })
                    if bl_correct is True or bl_correct == True:  # noqa: E712
                        fallback_correct += 1
                else:
                    no_baseline_count += 1
                    pq_rows.append({
                        "question_id": row["question_id"],
                        "model_name": model,
                        "benchmark": bm_label,
                        "method_name": method,
                        "original_parsed_choice": row.get("parsed_choice"),
                        "original_is_correct": row["is_correct"],
                        "baseline_parsed_choice": None,
                        "baseline_is_correct": None,
                        "fallback_used": False,
                        "fallback_is_correct": None,
                    })

            fallback_applied_correct = original_correct + fallback_correct
            fallback_applied_accuracy = fallback_applied_correct / total if total > 0 else 0.0
            fallback_gain = fallback_applied_accuracy - original_e2e

            notes = ""
            if no_baseline_count > 0:
                notes = f"{no_baseline_count}/{unscorable_count} unscorable rows had no matching baseline"

            summary_rows.append({
                "model_name": model,
                "benchmark": bm_label,
                "method_name": method,
                "total": total,
                "original_correct": original_correct,
                "original_end_to_end_accuracy": original_e2e,
                "original_scored": original_scored,
                "original_conditional_accuracy": original_cond,
                "unscorable_count": unscorable_count,
                "unscorable_rate": unscorable_rate,
                "fallback_available_count": fallback_available,
                "fallback_correct_count": fallback_correct,
                "fallback_applied_accuracy": fallback_applied_accuracy,
                "fallback_gain_vs_original": fallback_gain,
                "notes": notes,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(pq_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline fallback analysis. Writes to paper_results/reports_fallback/. "
            "Does not modify any input files."
        )
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Eval-ready folder containing the target method CSVs to analyze.",
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=None,
        help=(
            "Folder containing baseline rows. Defaults to --run-dir. "
            "Required when --run-dir is an ablation folder with no baseline rows "
            "(e.g. paper_api_ablation_v2). Point to paper_api_main or "
            "paper_combined_main."
        ),
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Filter to one benchmark: mmlu, arc, arc_challenge.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Override output directory. "
            f"Default: {_DEFAULT_OUTPUT_ROOT}/<folder_name>/<benchmark>/"
        ),
    )
    args = parser.parse_args()

    benchmark = _BENCHMARK_ALIASES.get(args.benchmark or "", args.benchmark)
    baseline_dir = args.baseline_dir or args.run_dir

    if not args.run_dir.exists():
        print(f"[error] --run-dir not found: {args.run_dir}")
        sys.exit(1)
    if not baseline_dir.exists():
        print(f"[error] --baseline-dir not found: {baseline_dir}")
        sys.exit(1)

    print(f"[fallback] Run dir   : {args.run_dir}")
    print(f"[fallback] Baseline  : {baseline_dir}")
    if benchmark:
        print(f"[fallback] Benchmark : {benchmark}")

    print("[fallback] Loading target method CSVs...")
    run_df = _load_csvs(args.run_dir, benchmark)
    print(f"[fallback]   {len(run_df)} rows loaded")

    if baseline_dir == args.run_dir:
        baseline_df = run_df
    else:
        print("[fallback] Loading baseline CSVs...")
        baseline_df = _load_csvs(baseline_dir, benchmark)
        print(f"[fallback]   {len(baseline_df)} rows loaded")

    # Hard check: baseline rows must be present
    bl_rows = baseline_df[baseline_df["method_name"] == "baseline"]
    if bl_rows.empty:
        print(
            "\n[error] No baseline rows found in the baseline source.\n"
            "        Fallback analysis requires matching baseline predictions.\n"
            "\n"
            "        If you are analyzing an ablation folder, use --baseline-dir\n"
            "        to point to a folder that contains baseline rows, e.g.:\n"
            "\n"
            "          --baseline-dir paper_results/eval_ready/paper_api_main\n"
            "          --baseline-dir paper_results/eval_ready/paper_combined_main\n"
        )
        sys.exit(1)

    bl_models = sorted(bl_rows["model_name"].dropna().unique())
    print(f"[fallback] Baseline models: {bl_models}")

    # Soft check: which target methods are actually present in run_df
    present_targets = set(run_df["method_name"].dropna().unique()) & set(FALLBACK_TARGET_METHODS)
    if not present_targets:
        print(
            f"\n[error] No fallback target methods found in {args.run_dir.name}.\n"
            f"        Expected one of: {FALLBACK_TARGET_METHODS}\n"
            f"        Found: {sorted(run_df['method_name'].dropna().unique())}\n"
        )
        sys.exit(1)
    print(f"[fallback] Target methods: {sorted(present_targets)}")

    # Build baseline index
    bl_index = _baseline_index(baseline_df)

    # Determine output path
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = _DEFAULT_OUTPUT_ROOT / args.run_dir.name
        if benchmark:
            out_dir = out_dir / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fallback] Output dir: {out_dir}")

    summary_df, pq_df = run_fallback_analysis(run_df, bl_index, benchmark)

    if summary_df.empty:
        print(
            "\n[warn] No results produced. "
            "Check that target methods and baseline models overlap.\n"
        )
        sys.exit(0)

    summary_path = out_dir / "fallback_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[fallback] Summary  -> {summary_path}")
    print(summary_df.to_string(index=False))

    if not pq_df.empty:
        pq_path = out_dir / "fallback_per_question.csv"
        pq_df.to_csv(pq_path, index=False)
        print(f"[fallback] Per-question -> {pq_path}")

    print("\n[fallback] Done.")


if __name__ == "__main__":
    main()
