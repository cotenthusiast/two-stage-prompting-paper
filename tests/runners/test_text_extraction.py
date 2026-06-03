# tests/runners/test_text_extraction.py

from pathlib import Path

import pytest

from twoprompt.runners.text_extraction import TextExtractionRunner
from twoprompt.scoring.types import SCORE_CORRECT, SCORE_INCORRECT, SCORE_UNSCORABLE
from twoprompt.parsing.types import PARSE_OK, PARSE_MISSING

from tests.runners.conftest import MockClient, _make_success_response, _make_failure_response

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = REPO_ROOT / "prompts"


def _make_runner(client):
    return TextExtractionRunner(
        client=client,
        method_name="text_extraction",
        split_name="robustness",
        prompt_version="v1",
        prompts_dir=_PROMPTS_DIR,
        run_id="test_run_001",
    )


class TestTextExtractionRunnerRunOne:
    """Tests for TextExtractionRunner.run_one execution flow."""

    @pytest.mark.asyncio
    async def test_correct_exact_match(self, runner_question_row, runner_metadata):
        """Model outputs 'HTTPS' — should match option C and score correct."""
        client = MockClient(responses=[_make_success_response("HTTPS", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "C"
        assert result["is_correct"] is True
        assert result["score_status"] == SCORE_CORRECT

    @pytest.mark.asyncio
    async def test_incorrect_match(self, runner_question_row, runner_metadata):
        """Model outputs 'FTP' — should match option A and score incorrect."""
        client = MockClient(responses=[_make_success_response("FTP", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] == "A"
        assert result["is_correct"] is False
        assert result["score_status"] == SCORE_INCORRECT

    @pytest.mark.asyncio
    async def test_unmatched_is_unscorable(self, runner_question_row, runner_metadata):
        """Model outputs gibberish — no option should match, result is unscorable."""
        client = MockClient(responses=[_make_success_response("quantum entanglement", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["score_status"] == SCORE_UNSCORABLE

    @pytest.mark.asyncio
    async def test_api_failure_returns_no_parse(self, runner_question_row, runner_metadata):
        """API failure — should return no parse or score."""
        client = MockClient(responses=[_make_failure_response(runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["parsed_choice"] is None
        assert result["is_correct"] is None
        assert result["score_status"] is None

    @pytest.mark.asyncio
    async def test_makes_exactly_one_api_call(self, runner_question_row, runner_metadata):
        """Text extraction should fire exactly one API call."""
        client = MockClient(responses=[_make_success_response("HTTPS", runner_metadata)])
        await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert len(client.requests_received) == 1

    @pytest.mark.asyncio
    async def test_prompt_includes_all_options(self, runner_question_row, runner_metadata):
        """The prompt sent to the model must include all four option texts."""
        client = MockClient(responses=[_make_success_response("HTTPS", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        prompt = result["prompt"]
        assert "FTP" in prompt
        assert "HTTP" in prompt
        assert "HTTPS" in prompt
        assert "SMTP" in prompt

    @pytest.mark.asyncio
    async def test_text_match_score_present(self, runner_question_row, runner_metadata):
        """Result row must include text_match_score."""
        client = MockClient(responses=[_make_success_response("HTTPS", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert "text_match_score" in result
        assert result["text_match_score"] is not None

    @pytest.mark.asyncio
    async def test_method_name_is_text_extraction(self, runner_question_row, runner_metadata):
        """method_name in result row must be text_extraction."""
        client = MockClient(responses=[_make_success_response("HTTPS", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["method_name"] == "text_extraction"

    @pytest.mark.asyncio
    async def test_raw_text_is_model_output(self, runner_question_row, runner_metadata):
        """raw_text should be the model-generated text, not a letter."""
        client = MockClient(responses=[_make_success_response("HTTPS", runner_metadata)])
        result = await _make_runner(client).run_one(runner_question_row, sample_index=0)

        assert result["raw_text"] == "HTTPS"
