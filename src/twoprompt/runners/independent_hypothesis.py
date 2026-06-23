# src/twoprompt/runners/independent_hypothesis.py

import asyncio
import math
import random
import re
from typing import Any

from twoprompt.parsing.types import PARSE_OK, ParseResult
from twoprompt.pipeline.prompt_builder import build_independent_hypothesis_prompt
from twoprompt.runners.base import ExperimentRunner

_LETTERS = ("A", "B", "C", "D")

# Matches "<score>X</score>" where X is an int or float, case-insensitive.
# Last occurrence wins, consistent with parser.py's last-occurrence philosophy
# for reasoning models that restate candidate values before the final one.
_SCORE_PATTERN = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)


class IndependentHypothesisRunner(ExperimentRunner):
    """Runner for the independent-hypothesis condition.

    Every real option for a question (4, or 3 for the rare ARC-Challenge
    question missing a D choice — ExperimentRunner._build_options never
    drops it, so this runner filters NaN/empty option text itself) is
    evaluated independently: the model receives the question and exactly
    one candidate option framed as a hypothesis, and returns a 0-100
    confidence score for that hypothesis alone. One parallel API call per
    real option (no option ever appears alongside another in the same
    prompt, and no call is ever made for a missing option). The final
    prediction is the option with the highest confidence score among the
    real options; ties are broken randomly using a seed derived from the
    run seed and question ID for reproducibility regardless of async
    completion order.

    All raw responses and scores are preserved in the result row so
    aggregation (e.g. a different tie-break or parse rule) can be redone
    post-hoc without rerunning inference.
    """

    async def run_one(self, question_row: Any, sample_index: int) -> dict:
        """Execute one question through the independent-hypothesis pipeline.

        Args:
            question_row: Normalized question record.
            sample_index: Repetition index for this question within the run.

        Returns:
            Flat result dictionary containing trace, model output, parse,
            and score fields, plus per-option raw text/score columns and
            the final prediction.
        """
        options = self._build_options(question_row)
        letters = [
            letter for letter in _LETTERS
            if not self._is_missing_option(options[letter])
        ]

        prompts = [
            build_independent_hypothesis_prompt(
                template=self._prompts["independent_hypothesis"],
                question=question_row["question_text"],
                option_text=options[letter],
            )
            for letter in letters
        ]
        requests = [
            self._build_model_request(question_row, prompt, sample_index)
            for prompt in prompts
        ]
        responses = await asyncio.gather(
            *[self.client.generate(req) for req in requests]
        )

        scores: dict[str, float] = {}
        parse_ok: dict[str, bool] = {}
        n_failures = 0
        for letter, response in zip(letters, responses):
            if response.is_success():
                score, ok = self._parse_confidence_score(response.raw_text)
            else:
                score, ok = 0.0, False
                n_failures += 1
            scores[letter] = score
            parse_ok[letter] = ok

        parsed_result = None
        score_result = None
        final_prediction = None

        if n_failures < len(letters):
            final_prediction = self._argmax_with_random_tiebreak(
                scores, self.seed, question_row["question_id"]
            )
            parsed_result = ParseResult(
                final_choice=final_prediction,
                status=PARSE_OK,
                raw_text=None,
                normalized_text="",
                reason="argmax_of_independent_hypothesis_scores",
            )
            score_result = self._score(parsed_result, question_row["correct_option"])

        result = self._build_result_row(
            question_row=question_row,
            prompt=prompts[0],
            model_request=requests[0],
            model_response=responses[0],
            parsed_result=parsed_result,
            score_result=score_result,
        )

        for letter, response in zip(letters, responses):
            suffix = letter.lower()
            result[f"option_{suffix}_raw_text"] = response.raw_text
            result[f"option_{suffix}_model_status"] = response.status
            result[f"option_{suffix}_score_parse_ok"] = parse_ok[letter]
            result[f"option_{suffix}_score"] = scores[letter]
        result["final_prediction"] = final_prediction
        result["n_model_failures"] = n_failures

        return result

    @staticmethod
    def _is_missing_option(value: Any) -> bool:
        """True if an option's text is NaN (CSV round-trip of an absent
        choice, e.g. ARC-Challenge questions with no real D option) or an
        empty/whitespace-only string.

        ExperimentRunner._build_options (shared by every runner) never
        drops a missing option, so this check happens here rather than
        upstream — changing the shared base would affect every method.
        """
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    @staticmethod
    def _parse_confidence_score(raw_text: str | None) -> tuple[float, bool]:
        """Extract a confidence score from raw model output via regex.

        Args:
            raw_text: Raw model output text, expected to contain a
                ``<score>X</score>`` tag.

        Returns:
            Tuple of (score, parse_ok). On parse failure, score is 0.0
            and parse_ok is False.
        """
        if not raw_text:
            return 0.0, False
        matches = _SCORE_PATTERN.findall(raw_text)
        if not matches:
            return 0.0, False
        try:
            return float(matches[-1]), True
        except ValueError:
            return 0.0, False

    @staticmethod
    def _argmax_with_random_tiebreak(
        scores: dict[str, float],
        seed: int | None,
        question_id: str,
    ) -> str:
        """Pick the highest-scoring option letter, breaking ties randomly.

        The tie-break RNG is seeded from (seed, question_id) rather than a
        shared mutable generator, so the outcome is reproducible regardless
        of the order in which concurrent questions complete.

        Args:
            scores: Mapping from option letter to confidence score.
            seed: Run seed for reproducible tie-breaking.
            question_id: Question identifier, mixed into the tie-break seed.

        Returns:
            The selected option letter.
        """
        best_score = max(scores.values())
        tied = sorted(letter for letter, s in scores.items() if s == best_score)
        if len(tied) == 1:
            return tied[0]
        rng = random.Random(f"{seed}:{question_id}")
        return rng.choice(tied)
