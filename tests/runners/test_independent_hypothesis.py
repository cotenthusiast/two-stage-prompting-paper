# tests/runners/test_independent_hypothesis.py

from pathlib import Path

import pytest

from twoprompt.runners.independent_hypothesis import IndependentHypothesisRunner
from twoprompt.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE

from tests.runners.conftest import MockClient, _make_success_response, _make_failure_response

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(client):
    return IndependentHypothesisRunner(
        client=client,
        method_name="independent_hypothesis",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
        seed=42,
    )


def _score_text(score) -> str:
    return f"Brief analysis. <score>{score}</score>"


class TestParseConfidenceScore:
    """Tests for IndependentHypothesisRunner._parse_confidence_score."""

    def test_parses_integer_score(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("text <score>72</score>")
        assert score == 72.0
        assert ok is True

    def test_parses_float_score(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("<score>12.5</score>")
        assert score == 12.5
        assert ok is True

    def test_missing_tag_falls_back_to_zero(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("no score tag here")
        assert score == 0.0
        assert ok is False

    def test_none_text_falls_back_to_zero(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score(None)
        assert score == 0.0
        assert ok is False

    def test_malformed_number_falls_back_to_zero(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score("<score>abc</score>")
        assert score == 0.0
        assert ok is False

    def test_last_occurrence_wins(self):
        score, ok = IndependentHypothesisRunner._parse_confidence_score(
            "<score>10</score> reconsidering... <score>90</score>"
        )
        assert score == 90.0
        assert ok is True


class TestArgmaxWithRandomTiebreak:
    """Tests for IndependentHypothesisRunner._argmax_with_random_tiebreak."""

    def test_clear_winner(self):
        scores = {"A": 10.0, "B": 90.0, "C": 5.0, "D": 0.0}
        assert IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, "q1") == "B"

    def test_tie_is_deterministic_for_same_seed_and_question(self):
        scores = {"A": 50.0, "B": 50.0, "C": 0.0, "D": 0.0}
        first = IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, "q1")
        second = IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, "q1")
        assert first == second
        assert first in {"A", "B"}

    def test_tie_can_differ_across_questions(self):
        scores = {"A": 50.0, "B": 50.0, "C": 0.0, "D": 0.0}
        results = {
            IndependentHypothesisRunner._argmax_with_random_tiebreak(scores, 42, f"q{i}")
            for i in range(20)
        }
        # With 20 different question IDs, both tied letters should appear at least once.
        assert results == {"A", "B"}


class TestIndependentHypothesisRunnerRunOne:
    """Tests for IndependentHypothesisRunner.run_one execution flow."""

    @pytest.mark.asyncio
    async def test_makes_four_independent_api_calls(self, runner_question_row, runner_metadata):
        responses = [_make_success_response(_score_text(50), runner_metadata) for _ in range(4)]
        client = MockClient(responses=responses)
        await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert len(client.requests_received) == 4

    @pytest.mark.asyncio
    async def test_each_prompt_contains_exactly_one_option(self, runner_question_row, runner_metadata):
        """Each independent call must see only its own option, never the others."""
        responses = [_make_success_response(_score_text(50), runner_metadata) for _ in range(4)]
        client = MockClient(responses=responses)
        await _make_runner(client).run_one(runner_question_row, sample_index=0)

        option_texts = ["FTP", "HTTP", "HTTPS", "SMTP"]
        for i, req in enumerate(client.requests_received):
            this_option = option_texts[i]
            # Exclude options that are substrings of this_option (e.g. "HTTP" in "HTTPS").
            other_options = [
                o for o in option_texts if o != this_option and o not in this_option
            ]
            assert this_option in req.payload
            for other in other_options:
                assert other not in req.payload

    @pytest.mark.asyncio
    async def test_highest_score_wins(self, runner_question_row, runner_metadata):
        """Option C (HTTPS, the correct answer) gets the highest score."""
        scores = [10, 20, 90, 5]  # A, B, C, D
        responses = [_make_success_response(_score_text(s), runner_metadata) for s in scores]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["final_prediction"] == "C"
        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    @pytest.mark.asyncio
    async def test_incorrect_prediction_scores_incorrect(self, runner_question_row, runner_metadata):
        scores = [95, 20, 10, 5]  # A wins, but correct answer is C
        responses = [_make_success_response(_score_text(s), runner_metadata) for s in scores]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["final_prediction"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    @pytest.mark.asyncio
    async def test_regex_parse_failure_falls_back_to_zero(self, runner_question_row, runner_metadata):
        """An option whose response has no <score> tag falls back to score 0."""
        responses = [
            _make_success_response("no tag here", runner_metadata),
            _make_success_response(_score_text(10), runner_metadata),
            _make_success_response(_score_text(20), runner_metadata),
            _make_success_response(_score_text(5), runner_metadata),
        ]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["option_a_score"] == 0.0
        assert result["option_a_score_parse_ok"] is False
        assert result["final_prediction"] == "C"  # highest of 0, 10, 20, 5

    @pytest.mark.asyncio
    async def test_single_call_failure_falls_back_to_zero_but_still_scores(
        self, runner_question_row, runner_metadata
    ):
        responses = [
            _make_failure_response(runner_metadata),
            _make_success_response(_score_text(10), runner_metadata),
            _make_success_response(_score_text(90), runner_metadata),
            _make_success_response(_score_text(5), runner_metadata),
        ]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["option_a_model_status"] == "failure"
        assert result["option_a_score"] == 0.0
        assert result["n_model_failures"] == 1
        assert result["final_prediction"] == "C"
        assert result["is_correct"] is True

    @pytest.mark.asyncio
    async def test_all_calls_fail_is_unscorable(self, runner_question_row, runner_metadata):
        responses = [_make_failure_response(runner_metadata) for _ in range(4)]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["n_model_failures"] == 4
        assert result["final_prediction"] is None
        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None

    @pytest.mark.asyncio
    async def test_per_option_columns_present(self, runner_question_row, runner_metadata):
        responses = [_make_success_response(_score_text(s), runner_metadata) for s in [1, 2, 3, 4]]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        for letter in ["a", "b", "c", "d"]:
            assert f"option_{letter}_raw_text" in result
            assert f"option_{letter}_model_status" in result
            assert f"option_{letter}_score_parse_ok" in result
            assert f"option_{letter}_score" in result
        assert result["option_a_score"] == 1.0
        assert result["option_b_score"] == 2.0
        assert result["option_c_score"] == 3.0
        assert result["option_d_score"] == 4.0

    @pytest.mark.asyncio
    async def test_method_name_is_independent_hypothesis(self, runner_question_row, runner_metadata):
        responses = [_make_success_response(_score_text(50), runner_metadata) for _ in range(4)]
        client = MockClient(responses=responses)
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["method_name"] == "independent_hypothesis"
