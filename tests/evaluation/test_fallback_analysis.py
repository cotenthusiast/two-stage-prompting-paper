# tests/evaluation/test_fallback_analysis.py
#
# Unit tests for scripts/fallback_analysis.py.
# Uses importlib to load the script (same pattern as test_evaluate_run.py).

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "fallback_analysis.py"
_spec = importlib.util.spec_from_file_location("fallback_analysis_script", _SCRIPT_PATH)
_fallback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fallback)

run_fallback_analysis = _fallback.run_fallback_analysis
_baseline_index = _fallback._baseline_index
FALLBACK_TARGET_METHODS = _fallback.FALLBACK_TARGET_METHODS

_EVAL_READY = Path(__file__).parents[2] / "paper_results" / "eval_ready"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    *,
    question_id: str,
    method_name: str,
    model_name: str = "gpt-4.1-mini",
    model_status: str = "success",
    is_correct=None,
    parsed_choice=None,
    benchmark: str = "mmlu",
) -> dict:
    return {
        "question_id": question_id,
        "method_name": method_name,
        "model_name": model_name,
        "model_status": model_status,
        "is_correct": is_correct,
        "parsed_choice": parsed_choice,
        "benchmark": benchmark,
    }


# ---------------------------------------------------------------------------
# FALLBACK_TARGET_METHODS coverage
# ---------------------------------------------------------------------------


class TestFallbackTargetMethods:
    def test_v2_v3_methods_in_targets(self):
        for m in ("two_prompt_v2", "two_prompt_v3", "twostage_semantic_match_v2"):
            assert m in FALLBACK_TARGET_METHODS, f"{m!r} missing from FALLBACK_TARGET_METHODS"

    def test_cyclic_not_in_targets(self):
        assert "cyclic" not in FALLBACK_TARGET_METHODS

    def test_pride_not_in_targets(self):
        assert "pride" not in FALLBACK_TARGET_METHODS

    def test_baseline_not_in_targets(self):
        assert "baseline" not in FALLBACK_TARGET_METHODS


# ---------------------------------------------------------------------------
# _baseline_index
# ---------------------------------------------------------------------------


class TestBaselineIndex:
    def test_returns_only_baseline_rows(self):
        df = pd.DataFrame([
            _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="A"),
            _row(question_id="q1", method_name="two_prompt", is_correct=None),
        ])
        idx = _baseline_index(df)
        assert len(idx) == 1
        assert ("gpt-4.1-mini", "q1") in idx.index

    def test_empty_when_no_baseline_rows(self):
        df = pd.DataFrame([
            _row(question_id="q1", method_name="two_prompt", is_correct=None),
        ])
        idx = _baseline_index(df)
        assert idx.empty


# ---------------------------------------------------------------------------
# run_fallback_analysis — core logic
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    return pd.DataFrame([
        _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C"),
        _row(question_id="q2", method_name="baseline", is_correct=False, parsed_choice="A"),
        # two_prompt: q1 unscorable, q2 scored
        _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
        _row(question_id="q2", method_name="two_prompt", is_correct=True, parsed_choice="C"),
    ])


class TestRunFallbackAnalysis:
    def test_summary_has_expected_columns(self, simple_df):
        bl_idx = _baseline_index(simple_df)
        summary, _ = run_fallback_analysis(simple_df, bl_idx, "mmlu")
        expected = {
            "model_name", "benchmark", "method_name", "total",
            "original_correct", "original_end_to_end_accuracy",
            "original_scored", "original_conditional_accuracy",
            "unscorable_count", "unscorable_rate",
            "fallback_available_count", "fallback_correct_count",
            "fallback_applied_accuracy", "fallback_gain_vs_original", "notes",
        }
        assert expected.issubset(set(summary.columns))

    def test_per_question_has_expected_columns(self, simple_df):
        bl_idx = _baseline_index(simple_df)
        _, pq = run_fallback_analysis(simple_df, bl_idx, "mmlu")
        expected = {
            "question_id", "model_name", "benchmark", "method_name",
            "original_parsed_choice", "original_is_correct",
            "baseline_parsed_choice", "baseline_is_correct",
            "fallback_used", "fallback_is_correct",
        }
        assert expected.issubset(set(pq.columns))

    def test_unscorable_count_is_correct(self, simple_df):
        bl_idx = _baseline_index(simple_df)
        summary, _ = run_fallback_analysis(simple_df, bl_idx, "mmlu")
        row = summary[summary["method_name"] == "two_prompt"].iloc[0]
        assert row["unscorable_count"] == 1

    def test_fallback_applied_accuracy_reflects_gain(self, simple_df):
        # q1 baseline is correct; q2 two_prompt is correct
        # original e2e = 1/2 = 0.5 (q1 unscorable=wrong, q2 correct)
        # after fallback: q1 takes baseline correct, q2 still correct → 2/2 = 1.0
        bl_idx = _baseline_index(simple_df)
        summary, _ = run_fallback_analysis(simple_df, bl_idx, "mmlu")
        row = summary[summary["method_name"] == "two_prompt"].iloc[0]
        assert abs(row["original_end_to_end_accuracy"] - 0.5) < 1e-9
        assert abs(row["fallback_applied_accuracy"] - 1.0) < 1e-9
        assert row["fallback_gain_vs_original"] > 0

    def test_per_question_fallback_used_for_unscorable(self, simple_df):
        bl_idx = _baseline_index(simple_df)
        _, pq = run_fallback_analysis(simple_df, bl_idx, "mmlu")
        q1_row = pq[pq["question_id"] == "q1"].iloc[0]
        assert bool(q1_row["fallback_used"]) is True

    def test_api_failure_rows_excluded_from_fallback(self):
        df = pd.DataFrame([
            _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C"),
            _row(question_id="q1", method_name="two_prompt", model_status="failure", is_correct=None),
        ])
        bl_idx = _baseline_index(df)
        summary, pq = run_fallback_analysis(df, bl_idx, "mmlu")
        row = summary[summary["method_name"] == "two_prompt"].iloc[0]
        assert row["unscorable_count"] == 0
        assert row["fallback_available_count"] == 0

    def test_no_baseline_in_index_produces_no_fallback_used(self):
        """If the baseline index is empty, no fallback_used rows are emitted."""
        df = pd.DataFrame([
            _row(question_id="q1", method_name="two_prompt", is_correct=None),
        ])
        bl_idx = _baseline_index(df)  # empty — no baseline rows
        summary, pq = run_fallback_analysis(df, bl_idx, "mmlu")
        if not pq.empty:
            assert not pq["fallback_used"].any()

    def test_models_do_not_cross_contaminate(self):
        df = pd.DataFrame([
            _row(question_id="q1", method_name="baseline", model_name="gpt-4.1-mini", is_correct=True, parsed_choice="C"),
            _row(question_id="q1", method_name="two_prompt", model_name="gemini-2.5-flash", is_correct=None),
        ])
        bl_idx = _baseline_index(df)
        summary, pq = run_fallback_analysis(df, bl_idx, "mmlu")
        gemini_row = summary[
            (summary["method_name"] == "two_prompt") &
            (summary["model_name"] == "gemini-2.5-flash")
        ]
        if not gemini_row.empty:
            assert gemini_row.iloc[0]["fallback_available_count"] == 0

    def test_empty_summary_when_no_target_methods(self):
        df = pd.DataFrame([
            _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C"),
            _row(question_id="q1", method_name="cyclic", is_correct=True, parsed_choice="C"),
        ])
        bl_idx = _baseline_index(df)
        summary, pq = run_fallback_analysis(df, bl_idx, "mmlu")
        assert summary.empty


# ---------------------------------------------------------------------------
# Smoke test against real eval_ready data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_EVAL_READY / "paper_api_main").exists(),
    reason="paper_results/eval_ready/paper_api_main not present",
)
class TestSmokeRealData:
    def test_paper_api_main_has_baseline_rows(self):
        from scripts.fallback_analysis import _load_csvs, _baseline_index
        df = _load_csvs(_EVAL_READY / "paper_api_main", "mmlu")
        idx = _baseline_index(df)
        assert not idx.empty

    def test_ablation_v2_has_no_baseline_rows(self):
        from scripts.fallback_analysis import _load_csvs, _baseline_index
        df = _load_csvs(_EVAL_READY / "paper_api_ablation_v2", "mmlu")
        idx = _baseline_index(df)
        assert idx.empty

    def test_fallback_runs_on_paper_api_main(self):
        from scripts.fallback_analysis import _load_csvs, _baseline_index, run_fallback_analysis
        df = _load_csvs(_EVAL_READY / "paper_api_main", "mmlu")
        idx = _baseline_index(df)
        summary, pq = run_fallback_analysis(df, idx, "mmlu")
        assert not summary.empty
        assert "fallback_applied_accuracy" in summary.columns
