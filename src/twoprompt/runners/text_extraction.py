# src/twoprompt/runners/text_extraction.py

from typing import Any

from twoprompt.parsing.text_matcher import TextMatchResult, match_text_to_options, parse_result_from_text_match
from twoprompt.pipeline.prompt_builder import build_text_extraction_prompt
from twoprompt.runners.base import ExperimentRunner
from twoprompt.scoring.scorer import score_prediction


class TextExtractionRunner(ExperimentRunner):
    """Runner for the text-extraction condition.

    The model sees the question AND all four options, but is instructed to
    output the answer text rather than a letter.  The generated text is then
    matched back to the closest option using normalized string matching and
    rapidfuzz fuzzy similarity (no second model call).

    This is a distinct experimental condition, not a two-stage ablation:
    - The model sees the options (unlike two-stage Stage 1).
    - The model outputs text, not a letter (unlike direct MCQ).
    - Matching is deterministic, not a second LLM call (unlike two-stage Stage 2).

    An extra ``text_match_score`` field is added to each result row.
    """

    async def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the text-extraction pipeline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary with standard fields plus text_match_score.
        """
        prompt = build_text_extraction_prompt(
            template=self._prompts["text_extraction"],
            question=question_row["question_text"],
            option_a=question_row["choice_a"],
            option_b=question_row["choice_b"],
            option_c=question_row["choice_c"],
            option_d=question_row["choice_d"],
        )
        request = self._build_model_request(question_row, prompt, sample_index)
        response = await self.client.generate(request)

        parsed_result = None
        score_result = None
        match: TextMatchResult | None = None

        if response.is_success():
            options = self._build_options(question_row)
            match = match_text_to_options(response.raw_text, options)
            parsed_result = parse_result_from_text_match(response.raw_text, options)
            score_result = score_prediction(parsed_result, question_row["correct_option"])

        result = self._build_result_row(
            question_row=question_row,
            prompt=prompt,
            model_request=request,
            model_response=response,
            parsed_result=parsed_result,
            score_result=score_result,
        )

        result["text_match_score"] = match.score if match is not None else None

        return result
