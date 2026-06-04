# tests/parsing/test_text_matcher.py

import math

import pytest

from twoprompt.parsing.text_matcher import _normalize, match_text_to_options, parse_result_from_text_match
from twoprompt.parsing.types import PARSE_OK, PARSE_MISSING

_OPTIONS = {
    "A": "FTP",
    "B": "HTTP",
    "C": "HTTPS",
    "D": "SMTP",
}

_OPTIONS_LONG = {
    "A": "Mitochondria",
    "B": "Nucleus",
    "C": "Ribosome",
    "D": "Cell membrane",
}


class TestNormalize:
    def test_normal_string(self):
        assert _normalize("HTTPS.") == "https"

    def test_none_returns_empty(self):
        assert _normalize(None) == ""

    def test_float_nan_returns_empty(self):
        assert _normalize(float("nan")) == ""

    def test_pandas_nan_float_returns_empty(self):
        # pandas reads missing CSV cells as float("nan") in object columns
        import pandas as pd
        nan_val = pd.Series(["a", None]).iloc[1]  # float nan from pandas
        assert _normalize(nan_val) == ""

    def test_non_string_non_nan_coerced(self):
        assert _normalize(42) == "42"


class TestMatchTextToOptions:
    def test_exact_match(self):
        result = match_text_to_options("HTTPS", _OPTIONS)
        assert result.label == "C"
        assert result.score == 1.0

    def test_exact_match_case_insensitive(self):
        result = match_text_to_options("https", _OPTIONS)
        assert result.label == "C"

    def test_exact_match_with_punctuation(self):
        result = match_text_to_options("HTTPS.", _OPTIONS)
        assert result.label == "C"

    def test_containment_match(self):
        result = match_text_to_options("The answer is Mitochondria", _OPTIONS_LONG)
        assert result.label == "A"

    def test_semantic_match(self):
        # "Mitocondria" misspelling — semantic similarity should still resolve to A
        result = match_text_to_options("Mitocondria", _OPTIONS_LONG)
        assert result.label == "A"

    def test_empty_answer_returns_no_match(self):
        result = match_text_to_options("", _OPTIONS)
        assert result.label is None

    def test_none_answer_returns_no_match(self):
        result = match_text_to_options(None, _OPTIONS)
        assert result.label is None

    def test_unmatched_returns_no_label(self):
        result = match_text_to_options("quantum entanglement", _OPTIONS)
        assert result.label is None

    def test_scores_by_option_always_present(self):
        result = match_text_to_options("HTTPS", _OPTIONS)
        assert set(result.scores_by_option.keys()) == {"A", "B", "C", "D"}

    def test_nan_answer_does_not_crash(self):
        result = match_text_to_options(float("nan"), _OPTIONS)
        assert result.label is None

    def test_none_answer_does_not_crash(self):
        result = match_text_to_options(None, _OPTIONS)
        assert result.label is None

    def test_nan_option_text_does_not_crash(self):
        options_with_nan = {
            "A": float("nan"),
            "B": "HTTP",
            "C": "HTTPS",
            "D": "SMTP",
        }
        result = match_text_to_options("HTTPS", options_with_nan)
        assert result.label == "C"

    def test_all_nan_options_does_not_crash(self):
        all_nan = {k: float("nan") for k in "ABCD"}
        result = match_text_to_options("HTTPS", all_nan)
        assert result.label is None


class TestParseResultFromTextMatch:
    def test_matched_gives_parse_ok(self):
        result = parse_result_from_text_match("HTTPS", _OPTIONS)
        assert result.status == PARSE_OK
        assert result.final_choice == "C"

    def test_unmatched_gives_parse_missing(self):
        result = parse_result_from_text_match("quantum entanglement", _OPTIONS)
        assert result.status == PARSE_MISSING
        assert result.final_choice is None

    def test_raw_text_preserved(self):
        raw = "HTTPS"
        result = parse_result_from_text_match(raw, _OPTIONS)
        assert result.raw_text == raw

    def test_reason_contains_score(self):
        result = parse_result_from_text_match("HTTPS", _OPTIONS)
        assert "cosine=" in result.reason
