"""Derive semantic-matching results from existing two-stage Stage 1 outputs.

This script re-uses the free-text Stage 1 responses already stored in
two_prompt run CSVs and replaces the second-stage LLM option-matching call
with deterministic text matching (exact normalisation, containment, then
sentence-transformers cosine similarity via text_matcher.match_text_to_options).

No model calls are made. Original two-stage files are not modified.

Purpose: test whether two-stage failures are caused by the second LLM
matching stage ("Did Stage 2 lose an answer that Stage 1 already had?").

Usage:
    python scripts/semantic_matching.py --run-id <RUN_ID> [--benchmark mmlu|arc_challenge]

Output:
    runs/<RUN_ID>/<RUN_ID>_twostage_semantic_match_<model>_<benchmark>.csv

Evaluate the output with the normal evaluate_run.py:
    python scripts/evaluate_run.py <RUN_ID> --benchmark mmlu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twoprompt.config.experiment import SEMANTIC_MATCH_METHOD, TWOPROMPT_METHOD
from twoprompt.parsing.text_matcher import match_text_to_options, parse_result_from_text_match
from twoprompt.scoring.scorer import score_prediction
from twoprompt.scoring.types import SCORE_UNSCORABLE

_BENCHMARK_ALIASES = {"arc": "arc_challenge"}

_OUTPUT_METHOD = SEMANTIC_MATCH_METHOD  # "twostage_semantic_match"


def _build_options(row: pd.Series) -> dict[str, str]:
    return {
        "A": row["choice_a"],
        "B": row["choice_b"],
        "C": row["choice_c"],
        "D": row["choice_d"],
    }


def derive_semantic_match(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Produce a semantic-match DataFrame from two_prompt rows.

    For each row:
    - If free_text_response is missing (Stage 1 failed), the row is carried
      forward as unscorable with model_status preserved.
    - Otherwise, match free_text_response to options deterministically and
      recompute parsed_choice / parse_status / is_correct / score_status.

    Args:
        df: DataFrame of two_prompt rows for one model+benchmark combination.
        benchmark: Benchmark name (for the benchmark column).

    Returns:
        New DataFrame with method_name = SEMANTIC_MATCH_METHOD.
    """
    out = df.copy()

    out["method_name"] = _OUTPUT_METHOD
    out["benchmark"] = benchmark

    parsed_choices = []
    parse_statuses = []
    normalized_texts = []
    parse_reasons = []
    is_corrects = []
    score_statuses = []
    text_match_scores = []

    for _, row in out.iterrows():
        ft = row.get("free_text_response")
        options = _build_options(row)

        if not isinstance(ft, str) or not ft.strip():
            parsed_choices.append(None)
            parse_statuses.append("parse_missing")
            normalized_texts.append(None)
            parse_reasons.append("free_text_response missing")
            is_corrects.append(None)
            score_statuses.append(SCORE_UNSCORABLE)
            text_match_scores.append(None)
            continue

        match = match_text_to_options(ft, options)
        parse_result = parse_result_from_text_match(ft, options)
        score_result = score_prediction(parse_result, row["correct_option"])

        parsed_choices.append(parse_result.final_choice)
        parse_statuses.append(parse_result.status)
        normalized_texts.append(parse_result.normalized_text)
        parse_reasons.append(parse_result.reason)
        is_corrects.append(score_result.is_correct)
        score_statuses.append(score_result.status)
        text_match_scores.append(match.score)

    out["parsed_choice"] = parsed_choices
    out["parse_status"] = parse_statuses
    out["normalized_text"] = normalized_texts
    out["parse_reason"] = parse_reasons
    out["is_correct"] = is_corrects
    out["score_status"] = score_statuses
    out["text_match_score"] = text_match_scores

    # raw_text becomes the free-text response (the text we matched from)
    out["raw_text"] = out["free_text_response"]

    return out


def find_two_prompt_files(run_dir: Path, benchmark: str | None) -> list[Path]:
    """Return all two_prompt CSV files for the given run and optional benchmark."""
    pattern = f"*_{TWOPROMPT_METHOD}_*.csv"
    files = sorted(run_dir.glob(pattern))
    if benchmark:
        files = [f for f in files if f.stem.endswith(f"_{benchmark}")]
    return files


def write_output(df: pd.DataFrame, run_dir: Path, run_id: str, model_name: str, benchmark: str) -> Path:
    """Write the derived CSV next to the original two_prompt files."""
    safe_model = model_name.replace("/", "_")
    fname = f"{run_id}_{_OUTPUT_METHOD}_{safe_model}_{benchmark}.csv"
    out_path = run_dir / fname
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive semantic-matching results from existing two-stage Stage 1 outputs."
    )
    parser.add_argument("--run-id", required=True, help="Run ID (folder name under runs/)")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Filter to a single benchmark (e.g. mmlu, arc_challenge, arc). "
             "Processes all two_prompt files if omitted.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "runs",
        help="Root runs directory (default: runs/)",
    )
    args = parser.parse_args()

    if args.benchmark:
        args.benchmark = _BENCHMARK_ALIASES.get(args.benchmark, args.benchmark)

    run_dir = args.runs_dir / args.run_id
    if not run_dir.is_dir():
        print(f"[error] Run directory not found: {run_dir}")
        sys.exit(1)

    files = find_two_prompt_files(run_dir, args.benchmark)
    if not files:
        spec = f"benchmark={args.benchmark}" if args.benchmark else "any benchmark"
        print(f"[error] No two_prompt CSV files found in {run_dir} for {spec}")
        sys.exit(1)

    print(f"[semantic_matching] Run: {args.run_id}")
    print(f"[semantic_matching] Found {len(files)} two_prompt file(s) to process")

    for csv_path in files:
        df = pd.read_csv(csv_path)
        if "free_text_response" not in df.columns:
            print(f"[skip] {csv_path.name} — no free_text_response column")
            continue

        # Infer model and benchmark from the original DataFrame columns
        model_names = df["model_name"].dropna().unique()
        benchmarks = df["benchmark"].dropna().unique() if "benchmark" in df.columns else []
        if len(model_names) != 1:
            print(f"[skip] {csv_path.name} — unexpected model_name values: {model_names}")
            continue

        model_name = model_names[0]
        benchmark = benchmarks[0] if len(benchmarks) == 1 else (args.benchmark or "unknown")

        print(f"[processing] {csv_path.name}  ({len(df)} rows, model={model_name}, benchmark={benchmark})")

        derived = derive_semantic_match(df, benchmark)

        # Sanity: confirm no model calls were made (row count unchanged)
        assert len(derived) == len(df), "Row count changed unexpectedly"

        out_path = write_output(derived, run_dir, args.run_id, model_name, benchmark)
        n_scorable = int(derived["is_correct"].notna().sum())
        n_correct = int(derived["is_correct"].eq(True).sum())
        pct = 100 * n_correct / len(derived) if len(derived) > 0 else 0.0
        print(f"  → {out_path.name}  correct={n_correct}/{n_scorable} scorable  e2e={pct:.1f}%")

    print("[semantic_matching] Done. Evaluate with:")
    bench_arg = f" --benchmark {args.benchmark}" if args.benchmark else ""
    print(f"  python scripts/evaluate_run.py {args.run_id}{bench_arg}")


if __name__ == "__main__":
    main()
