# tests/evaluation/test_evaluate_run.py
#
# Tests for evaluate_run.py: METHOD_ORDER coverage, EXTERNALLY_SCORED_METHODS,
# reparse_run exclusions, and baseline fallback logic.
# The script lives outside the twoprompt package, so we load it via importlib.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "evaluate_run.py"
_spec = importlib.util.spec_from_file_location("evaluate_run_script", _SCRIPT_PATH)
_evaluate_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate_run)

apply_baseline_fallback = _evaluate_run.apply_baseline_fallback
compute_accuracy = _evaluate_run.compute_accuracy
reparse_run = _evaluate_run.reparse_run
METHOD_ORDER = _evaluate_run.METHOD_ORDER


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
    parse_status: str | None = None,
    correct_option: str = "C",
    subject: str = "anatomy",
) -> dict:
    return {
        "question_id": question_id,
        "method_name": method_name,
        "model_name": model_name,
        "model_status": model_status,
        "is_correct": is_correct,
        "parsed_choice": parsed_choice,
        "parse_status": parse_status,
        "correct_option": correct_option,
        "subject": subject,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_mixed():
    """A DataFrame with baseline rows and unscorable two-stage rows."""
    return pd.DataFrame(
        [
            # Baseline — one correct, one incorrect
            _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C", parse_status="ok"),
            _row(question_id="q2", method_name="baseline", is_correct=False, parsed_choice="A", parse_status="ok"),
            # two_prompt — q1 unscorable (parse failure), q2 scored normally
            _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None, parse_status="missing"),
            _row(question_id="q2", method_name="two_prompt", is_correct=True, parsed_choice="C", parse_status="ok"),
            # two_prompt_cyclic — q1 unscorable
            _row(question_id="q1", method_name="two_prompt_cyclic", is_correct=None, parsed_choice=None, parse_status="missing"),
            # cyclic — q1 unscorable (should NOT be substituted)
            _row(question_id="q1", method_name="cyclic", is_correct=None, parsed_choice=None, parse_status="missing"),
        ]
    )


# ---------------------------------------------------------------------------
# apply_baseline_fallback — column injection
# ---------------------------------------------------------------------------


class TestApplyBaselineFallbackColumn:
    def test_fallback_applied_column_is_added(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        assert "fallback_applied" in result.columns

    def test_fallback_applied_has_no_null_values(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        assert result["fallback_applied"].notna().all()

    def test_original_df_not_mutated(self, df_mixed):
        original_nulls = df_mixed["is_correct"].isna().sum()
        apply_baseline_fallback(df_mixed)
        assert df_mixed["is_correct"].isna().sum() == original_nulls


# ---------------------------------------------------------------------------
# apply_baseline_fallback — substitution logic
# ---------------------------------------------------------------------------


class TestApplyBaselineFallbackSubstitution:
    def test_unscorable_two_prompt_row_gets_is_correct_from_baseline(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["is_correct"] is True

    def test_unscorable_two_prompt_row_gets_parsed_choice_from_baseline(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["parsed_choice"] == "C"

    def test_unscorable_two_prompt_cyclic_row_substituted(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt_cyclic")]
        assert bool(row.iloc[0]["is_correct"]) is True
        assert bool(row.iloc[0]["fallback_applied"]) is True

    def test_already_scored_row_not_changed(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q2") & (result["method_name"] == "two_prompt")]
        assert bool(row.iloc[0]["is_correct"]) is True
        assert bool(row.iloc[0]["fallback_applied"]) is False

    def test_baseline_rows_never_marked_as_fallback(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        bl = result[result["method_name"] == "baseline"]
        assert bl["fallback_applied"].eq(False).all()

    def test_cyclic_rows_not_substituted(self, df_mixed):
        result = apply_baseline_fallback(df_mixed)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "cyclic")]
        assert row.iloc[0]["is_correct"] is None
        assert bool(row.iloc[0]["fallback_applied"]) is False


# ---------------------------------------------------------------------------
# apply_baseline_fallback — eligibility rules
# ---------------------------------------------------------------------------


class TestApplyBaselineFallbackEligibility:
    def test_api_failure_rows_are_not_substituted(self):
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="two_prompt", model_status="failure", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["is_correct"] is None
        assert bool(row.iloc[0]["fallback_applied"]) is False

    def test_no_baseline_means_no_substitution(self):
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        assert result.iloc[0]["is_correct"] is None

    def test_question_id_must_match_for_substitution(self):
        """Baseline for q2 must not bleed into an unscorable q1 row."""
        df = pd.DataFrame(
            [
                _row(question_id="q2", method_name="baseline", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert row.iloc[0]["is_correct"] is None

    def test_different_models_do_not_cross_contaminate(self):
        """Baseline from gpt-4.1-mini must not substitute into a gemini row."""
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", model_name="gpt-4.1-mini", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="two_prompt", model_name="gemini-2.5-flash", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["model_name"] == "gemini-2.5-flash")]
        assert row.iloc[0]["is_correct"] is None

    def test_fallback_transfers_incorrect_baseline_result(self):
        """Baseline answer can be wrong; the fallback still uses whatever the baseline had."""
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", is_correct=False, parsed_choice="A", parse_status="ok"),
                _row(question_id="q1", method_name="two_prompt", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        row = result[(result["question_id"] == "q1") & (result["method_name"] == "two_prompt")]
        assert bool(row.iloc[0]["is_correct"]) is False
        assert row.iloc[0]["parsed_choice"] == "A"
        assert bool(row.iloc[0]["fallback_applied"]) is True

    def test_multiple_models_each_fallback_independently(self):
        df = pd.DataFrame(
            [
                _row(question_id="q1", method_name="baseline", model_name="gpt-4.1-mini", is_correct=True, parsed_choice="C"),
                _row(question_id="q1", method_name="baseline", model_name="gemini-2.5-flash", is_correct=False, parsed_choice="A"),
                _row(question_id="q1", method_name="two_prompt", model_name="gpt-4.1-mini", is_correct=None, parsed_choice=None),
                _row(question_id="q1", method_name="two_prompt", model_name="gemini-2.5-flash", is_correct=None, parsed_choice=None),
            ]
        )
        result = apply_baseline_fallback(df)
        gpt_row = result[(result["model_name"] == "gpt-4.1-mini") & (result["method_name"] == "two_prompt")]
        gemini_row = result[(result["model_name"] == "gemini-2.5-flash") & (result["method_name"] == "two_prompt")]
        assert bool(gpt_row.iloc[0]["is_correct"]) is True
        assert bool(gemini_row.iloc[0]["is_correct"]) is False


# ---------------------------------------------------------------------------
# compute_accuracy — fallback_count column
# ---------------------------------------------------------------------------


class TestComputeAccuracyFallbackCount:
    def test_fallback_count_zero_when_no_fallback_applied(self, df_mixed):
        """Without applying fallback, fallback_count should be 0 for all rows."""
        # df_mixed has no fallback_applied column — count should default to 0
        accuracy = compute_accuracy(df_mixed)
        assert "fallback_count" in accuracy.columns
        assert (accuracy["fallback_count"] == 0).all()

    def test_fallback_count_reflects_substitutions(self, df_mixed):
        """After fallback, the count should match actual substitutions."""
        df_after = apply_baseline_fallback(df_mixed)
        accuracy = compute_accuracy(df_after)
        # q1 was substituted in two_prompt — fallback_count for (two_prompt, gpt-4.1-mini) == 1
        row = accuracy[(accuracy["method"] == "two_prompt") & (accuracy["model"] == "gpt-4.1-mini")]
        assert row.iloc[0]["fallback_count"] == 1

    def test_fallback_count_zero_for_baseline_method(self, df_mixed):
        df_after = apply_baseline_fallback(df_mixed)
        accuracy = compute_accuracy(df_after)
        bl = accuracy[accuracy["method"] == "baseline"]
        assert (bl["fallback_count"] == 0).all()


# ---------------------------------------------------------------------------
# METHOD_ORDER coverage — v2/v3 ablation method names
# ---------------------------------------------------------------------------


class TestMethodOrder:
    def test_two_prompt_v2_in_method_order(self):
        assert "two_prompt_v2" in METHOD_ORDER

    def test_two_prompt_v3_in_method_order(self):
        assert "two_prompt_v3" in METHOD_ORDER

    def test_twostage_semantic_match_v2_in_method_order(self):
        assert "twostage_semantic_match_v2" in METHOD_ORDER

    def test_core_methods_still_present(self):
        for m in ("baseline", "two_prompt", "cyclic", "pride", "text_extraction", "twostage_semantic_match"):
            assert m in METHOD_ORDER, f"{m!r} missing from METHOD_ORDER"

    def test_method_order_has_no_duplicates(self):
        assert len(METHOD_ORDER) == len(set(METHOD_ORDER))


# ---------------------------------------------------------------------------
# reparse_run — externally scored methods must not be reparsed
# ---------------------------------------------------------------------------


def _reparsable_df(method_name: str) -> pd.DataFrame:
    """Minimal DataFrame for testing reparse_run exclusions."""
    return pd.DataFrame([{
        "question_id": "q1",
        "method_name": method_name,
        "model_name": "gpt-4.1-mini",
        "model_status": "success",
        "parse_reason": "initial",
        "raw_text": "The answer is A.",
        "parsed_choice": None,
        "parse_status": None,
        "normalized_text": None,
        "is_correct": None,
        "score_status": None,
        "correct_option": "A",
        "choice_a": "alpha",
        "choice_b": "beta",
        "choice_c": "gamma",
        "choice_d": "delta",
    }])


class TestReparseRunExclusions:
    def test_twostage_semantic_match_v2_not_reparsed(self):
        df = _reparsable_df("twostage_semantic_match_v2")
        result = reparse_run(df)
        # parsed_choice should remain None — row was not reparsed
        assert result.iloc[0]["parsed_choice"] is None or pd.isna(result.iloc[0]["parsed_choice"])

    def test_twostage_semantic_match_not_reparsed(self):
        df = _reparsable_df("twostage_semantic_match")
        result = reparse_run(df)
        assert result.iloc[0]["parsed_choice"] is None or pd.isna(result.iloc[0]["parsed_choice"])

    def test_pride_not_reparsed(self):
        df = _reparsable_df("pride")
        result = reparse_run(df)
        assert result.iloc[0]["parsed_choice"] is None or pd.isna(result.iloc[0]["parsed_choice"])

    def test_text_extraction_not_reparsed(self):
        df = _reparsable_df("text_extraction")
        result = reparse_run(df)
        assert result.iloc[0]["parsed_choice"] is None or pd.isna(result.iloc[0]["parsed_choice"])

    def test_two_prompt_is_reparsed(self):
        """two_prompt is not externally scored — it should go through reparse_run."""
        df = _reparsable_df("two_prompt")
        result = reparse_run(df)
        # "The answer is A." should parse to A
        assert result.iloc[0]["parsed_choice"] == "A"

    def test_two_prompt_v2_is_reparsed(self):
        df = _reparsable_df("two_prompt_v2")
        result = reparse_run(df)
        assert result.iloc[0]["parsed_choice"] == "A"

    def test_two_prompt_v3_is_reparsed(self):
        df = _reparsable_df("two_prompt_v3")
        result = reparse_run(df)
        assert result.iloc[0]["parsed_choice"] == "A"
